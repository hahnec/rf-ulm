import logging
import os
import random
import sys
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from omegaconf import OmegaConf

from datasets.pala_dataset.pala_iq import PalaDatasetIq
from datasets.pala_dataset.pala_rf import PalaDatasetRf
from models.unet import UNet, SlounUNet, SlounAdaptUNet
from models.mspcn import MSPCN
from models.edsr import EDSR
from models.smv.lstm_unet import UNet_ConvLSTM
from evaluate import evaluate
from utils.nms_funs import non_max_supp_torch
from utils.gauss import matlab_style_gauss2D
from utils.dice_score import dice_loss
from utils.transform import ArgsToTensor, NormalizeImage, NormalizeVol, RandomHorizontalFlip, RandomVerticalFlip, RandomCropScale, RandomRotation, RandomApply
from utils.samples_points_map import get_inverse_mapping


img_norm = lambda x: (x-x.min())/(x.max()-x.min()) if (x.max()-x.min()) != 0 else x


def train_model(
        model,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        cfg = None,
):
    # create dataset
    scale_factor = 1 if cfg.model in ('unet') else cfg.upscale_factor
    crop_size = 128 if cfg.input_type == 'rf' else 64
    crop_size = crop_size * cfg.upscale_factor if cfg.model in ('unet') else crop_size
    if cfg.input_type == 'iq':
        DatasetClass = PalaDatasetIq
        rand_augment = RandomApply([RandomHorizontalFlip(), RandomVerticalFlip(), RandomRotation(5)])
        transforms = [ArgsToTensor(), rand_augment, RandomCropScale(crop_size, scale_factor), NormalizeImage()] 
        from datasets.pala_dataset.utils.collate_fn_iq import collate_fn
    elif cfg.input_type == 'rf':
        DatasetClass = PalaDatasetRf
        rand_augment = RandomApply([RandomVerticalFlip(), RandomRotation(5)])
        transforms = [ArgsToTensor(), rand_augment, RandomCropScale(crop_size, scale_factor), NormalizeVol()]
        from datasets.pala_dataset.utils.collate_fn_rf import collate_fn
    dataset = DatasetClass(
        dataset_path = cfg.data_dir,
        train = True,
        transforms = transforms,
        clutter_db = cfg.clutter_db,
        sequences = [15, 16, 17, 18, 19] if not cfg.data_dir.lower().__contains__('home') else cfg.sequences,
        rescale_factor = cfg.rescale_factor,
        upscale_factor = cfg.upscale_factor,
        transducer_interp = True,
        temporal_filter_opt = cfg.data_dir.lower().__contains__('rat'),
        tile_opt = cfg.model in ('unet', 'smv'),
        scale_opt = cfg.model in ('unet', 'smv'),
        angle_threshold = cfg.angle_threshold,
        )

    # data-related configuration
    cfg.wavelength = float(dataset.get_key('wavelength'))
    cfg.origin_x = float(dataset.get_key('Origin')[0])
    cfg.origin_z = float(dataset.get_key('Origin')[2])
    cfg.wv_idcs = list(range(3)) if cfg.wv_idcs is None else cfg.wv_idcs
    cfg.wv_idcs = [0] if cfg.input_type == 'iq' else cfg.wv_idcs

    # split into train and validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    division_step = (n_train // (5 * batch_size))

    # create data loaders
    num_workers = min(4, os.cpu_count())
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, collate_fn=collate_fn, shuffle=False if cfg.model == 'smv' else True, **loader_args)
    val_loader = DataLoader(val_set, collate_fn=collate_fn, shuffle=False, drop_last=True, **loader_args)

    # instantiate logging
    if cfg.logging:
        wb = wandb.init(project='SR-ULM-TRAIN', resume='allow', anonymous='must', config=cfg, group='train')
        wb.config.update(
            dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, val_percent=val_percent, img_scale=img_scale, amp=amp)
        )
        wandb.define_metric('epoch', step_metric='epoch')
        wandb.define_metric('train_loss', step_metric='train_step')
        wandb.define_metric('threshold', step_metric='val_step')
        wandb.define_metric('avg_detected', step_metric='val_step')
        wandb.define_metric('pred_max', step_metric='val_step')
        wandb.define_metric('lr', step_metric='epoch')
        wandb.define_metric('validation_dice', step_metric='val_step')
        wandb.define_metric('images', step_metric='val_step')
        wandb.define_metric('masks', step_metric='val_step')

        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Device:          {cfg.device}
            Images scaling:  {img_scale}
            Mixed Precision: {amp}
        ''')

    # set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    #optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, foreach=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
    #scheduler = optim.lr_scheduler.PolynomialLR(optimizer, cfg.epochs, power=1)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.MSELoss(reduction='mean')
    l1loss = nn.L1Loss(reduction='mean')
    lambda_value = 0.01 if cfg.model in ('unet', 'smv') and cfg.input_type == 'iq' else cfg.lambda1
    train_step = 0
    val_step = 0
    
    # mSPCN Gaussian
    g_len = 7+cfg.upscale_factor//2*2
    psf_heatmap = torch.from_numpy(matlab_style_gauss2D(shape=(g_len,g_len), sigma=cfg.upscale_factor/4))
    gfilter = torch.reshape(psf_heatmap, [1, 1, g_len, g_len])
    gfilter = gfilter.to(cfg.device)
    if cfg.model.__contains__('mspcn') and cfg.input_type == 'iq':
        amplitude = 50
    elif cfg.model in ('unet', 'smv') and cfg.input_type == 'iq':
        amplitude = 1
    else: 
        amplitude = cfg.lambda0

    # transformation
    t_mats = get_inverse_mapping(dataset, p=6, weights_opt=False, point_num=1e4) if cfg.input_type == 'rf' else [[],[],[]]

    # training
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs, true_masks = batch[:2] if cfg.input_type == 'iq' else (batch[0].flatten(0, 1), batch[1].flatten(0, 1))

                # skip blank frames (avoid learning from false frames)
                if torch.any(imgs.view(imgs.shape[0], -1).sum() == 0) or torch.any(true_masks.view(true_masks.shape[0], -1).sum() == 0):
                    continue

                imgs = imgs.to(device=cfg.device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=cfg.device, dtype=torch.float32)

                with torch.autocast(cfg.device if cfg.device != 'mps' else 'cpu', enabled=amp):
                    
                    # use batch size (without shuffling) for temporal stacking with new batch size 1
                    if cfg.model == 'smv': 
                        imgs = imgs.unsqueeze(0)
                        true_masks = true_masks.sum(0, keepdim=True)

                    pred_masks = model(imgs)

                    # mask blurring
                    true_masks = F.conv2d(true_masks, gfilter, padding=gfilter.shape[-1]//2)
                    true_masks /= true_masks.max()
                    true_masks *= amplitude
                    if cfg.model == 'mspcn' and cfg.input_type == 'iq':
                        pred_masks = F.conv2d(pred_masks, gfilter, padding=gfilter.shape[-1]//2)
                        
                    loss = criterion(pred_masks.squeeze(1), true_masks.squeeze(1).float())
                    loss += l1loss(pred_masks.squeeze(1), torch.zeros_like(pred_masks.squeeze(1))) * lambda_value

                #mask = masks_true[0, 0, ::cfg.upscale_factor, ::cfg.upscale_factor] * amplitude
                #img = images[0, 0].clone()
                #img[img<0] = 0
                #img = img/img.max()
                #mask = mask/mask.max()
                #mask_review_img = torch.dstack([mask, img, mask]).cpu().numpy()
                #import matplotlib.pyplot as plt
                #fig, axs = plt.subplots(nrows=1,ncols=1, figsize=(15,9))
                ##axs[0].imshow(images[0, 0].cpu().numpy())
                ##axs[1].imshow(masks_true[0].cpu().numpy())
                #axs.imshow(mask_review_img)
                #import imageio
                #imageio.imsave('rf_label_frame.png', mask_review_img)
                #plt.show()

                # activation followed by non-maximum suppression
                #masks_pred = torch.sigmoid(masks_pred)
                #imgs_nms = non_max_supp_torch(masks_pred)
                #masks_nms = imgs_nms > cfg.nms_threshold

                optimizer.zero_grad(set_to_none=True)
                scale = grad_scaler.get_scale()
                grad_scaler.update()
                skip_lr_schedule = scale > grad_scaler.get_scale()

                if not skip_lr_schedule:
                    grad_scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    grad_scaler.step(optimizer)

                pbar.update(imgs.shape[0])
                train_step += 1
                epoch_loss += loss.item()
                if cfg.logging:
                    wb.log({
                        'train loss': loss.item(),
                        'train_step': train_step,
                        'epoch': epoch
                    })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # evaluation
                if train_step % division_step == 0 and division_step > 0 and not skip_lr_schedule:
                    histograms = {}
                    for tag, value in model.named_parameters():
                        tag = tag.replace('/', '.')
                        if not torch.isinf(value).any() and not torch.isnan(value).any():
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        if not torch.isinf(value.grad).any() and not torch.isnan(value.grad).any():
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                    
                    val_score, pala_err_batch, masks_nms, threshold = evaluate(model, val_loader, amp, cfg, t_mats)

                    logging.info('Validation Dice score: {}'.format(val_score))
                    try:
                        if cfg.logging:
                            wb.log({
                                'lr': optimizer.param_groups[0]['lr'],
                                'validation_dice': val_score,
                                'images': wandb.Image(imgs[0].cpu() if len(imgs[0]) == 2 else imgs[0].sum(0).cpu()),
                                'masks': {
                                    'true': wandb.Image(img_norm(true_masks[0].float().cpu())*255),
                                    'pred': wandb.Image(img_norm(pred_masks[0].float().cpu())*255),    #(masks_pred.argmax(dim=1)[0]).float().cpu()),#
                                    'nms': wandb.Image(img_norm(masks_nms[0].float().cpu())*255),
                                },
                                'val_step': val_step,
                                'epoch': epoch,
                                'threshold': threshold,
                                'avg_detected': float(masks_nms[0].float().cpu().sum()),
                                'pred_max': float(pred_masks[0].float().cpu().max()),
                                **histograms
                            })
                    except Exception as e:
                        print('Validation upload failed')
                        print(e)
                    val_step += 1
    
        scheduler.step()

    if cfg.logging:
        dir_checkpoint = Path('./ckpts/')
        dir_checkpoint.mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        torch.save(state_dict, str(dir_checkpoint / (wb.name+str('_ckpt_epoch{}.pth'.format(epoch)))))
        logging.info(f'Checkpoint {epoch} saved!')
        wandb.finish()


if __name__ == '__main__':


    # load configuration
    cfg = OmegaConf.load('./config.yml')

    # override loaded configuration with CLI arguments
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    
    # for reproducibility
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {cfg.device}')

    # model selection
    in_channels = 1 if cfg.input_type == 'rf' and cfg.rescale_factor != 1 else 2
    if cfg.model == 'unet':
        # U-Net
        model = SlounAdaptUNet(n_channels=in_channels, n_classes=1, bilinear=False)
    elif cfg.model == 'mspcn':
        # mSPCN
        model = MSPCN(upscale_factor=cfg.upscale_factor, in_channels=in_channels)
    elif cfg.model == 'edsr':
        # EDSR
        class Args:
            pass
        args = Args()
        args.n_feats = 64
        args.n_resblocks = 16
        args.n_colors = in_channels
        args.rgb_range = 1
        args.scale = (cfg.upscale_factor, cfg.upscale_factor)
        args.res_scale = 1
        model = EDSR(args)
    elif cfg.model == 'smv':
        model = UNet_ConvLSTM(n_channels=in_channels, n_classes=1, use_LSTM=True, parallel_encoder=False, lstm_layers=1)
    else:
        raise Exception('Model name not recognized')

    model = model.to(memory_format=torch.channels_last)

    if cfg.fine_tune:
        ckpt_paths = [fn for fn in Path('./ckpts').iterdir() if fn.name.startswith(cfg.model_file.split('_')[0])]
        state_dict = torch.load(str(ckpt_paths[0]), map_location=cfg.device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {cfg.model_file}')

    model.to(device=cfg.device)
    train_model(
        model=model,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        img_scale=0.5,
        val_percent=0.1,
        amp=False,
        cfg = cfg
    )
