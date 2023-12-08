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
from evaluate import evaluate
from utils.nms_funs import non_max_supp_torch
from utils.gauss import matlab_style_gauss2D
from utils.dice_score import dice_loss
from utils.transform import ArgsToTensor, NormalizeImage, NormalizeVol, RandomHorizontalFlip, RandomVerticalFlip, RandomCropScale, GaussianBlur, RandomRotation, RandomApply
from utils.samples_points_map import get_inverse_mapping
from utils.threshold import estimate_threshold


def train_model(
        model,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        cfg = None,
):
    # create dataset
    scale_factor = 1 if cfg.model in ('unet') else cfg.upscale_factor
    crop_size = 64 if cfg.input_type == 'iq' else 128
    crop_size = crop_size * cfg.upscale_factor if cfg.model in ('unet') else crop_size
    if cfg.input_type == 'iq':
        DatasetClass = PalaDatasetIq
        rand_augment = RandomApply([RandomHorizontalFlip(), RandomVerticalFlip(), GaussianBlur(5, (.6, .4)), RandomRotation(5)])
        transforms = [ArgsToTensor(), rand_augment, RandomCropScale(crop_size, scale_factor), NormalizeImage()] 
        from datasets.pala_dataset.utils.collate_fn_iq import collate_fn
    elif cfg.input_type == 'rf':
        DatasetClass = PalaDatasetRf
        Normalizer = NormalizeVol if cfg.skip_bmode else NormalizeImage
        rand_augment = RandomApply([RandomVerticalFlip(), GaussianBlur(5, (.6, .4)), RandomRotation(5)])
        transforms = [ArgsToTensor(), rand_augment, RandomCropScale(crop_size, scale_factor), Normalizer()]
        from datasets.pala_dataset.utils.collate_fn_rf import collate_fn
    dataset = DatasetClass(
        dataset_path = cfg.data_dir,
        train = True,
        transforms = transforms,
        clutter_db = cfg.clutter_db,
        sequences = [15, 16, 17, 18, 19] if not cfg.data_dir.lower().__contains__('home') else cfg.sequences,
        rescale_factor = cfg.rescale_factor,
        upscale_factor = cfg.upscale_factor,
        upscale_channels = cfg.channel_num,
        transducer_interp = True,
        temporal_filter_opt = cfg.data_dir.lower().__contains__('rat'),
        tile_opt = cfg.model in ('unet'),
        scale_opt = cfg.model in ('unet'),
        angle_threshold = cfg.angle_threshold,
        )

    # data-related configuration
    cfg.wavelength = float(dataset.get_key('wavelength'))
    cfg.origin_x = float(dataset.get_key('Origin')[0])
    cfg.origin_z = float(dataset.get_key('Origin')[2])
    cfg.wv_idcs = list(range(3)) if cfg.wv_idcs is None else cfg.wv_idcs
    cfg.wv_idcs = [0] if cfg.input_type == 'iq' else cfg.wv_idcs
    cfg.nms_size = cfg.upscale_factor if cfg.nms_size is None else cfg.nms_size
    batch_size = 8 if batch_size > 8 and cfg.model in ('unet') else batch_size

    # split into train and validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))
    division_step = (n_train // (5 * batch_size))

    # create data loaders
    num_workers = min(4, os.cpu_count())
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, collate_fn=collate_fn, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, collate_fn=collate_fn, shuffle=False, drop_last=True, **loader_args)

    # instantiate logging
    wb = None
    if cfg.logging:
        wb = wandb.init(project='SR-ULM-TRAIN', resume='allow', anonymous='must', config=cfg, group='train')
        wb.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, val_percent=val_percent, amp=amp))
        wandb.define_metric('epoch', step_metric='epoch')
        wandb.define_metric('train_loss', step_metric='train_step')
        wandb.define_metric('val_loss', step_metric='val_step')
        wandb.define_metric('threshold', step_metric='val_step')
        wandb.define_metric('avg_detected', step_metric='epoch')
        wandb.define_metric('pred_max', step_metric='epoch')
        wandb.define_metric('lr', step_metric='epoch')
        wandb.define_metric('validation_dice', step_metric='epoch')
        wandb.define_metric('images', step_metric='epoch')
        wandb.define_metric('masks', step_metric='epoch')

        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Device:          {cfg.device}
            Mixed Precision: {amp}
        ''')

    # set up the optimizer, the loss, the learning rate scheduler and the loss scaling
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, foreach=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    mse_loss = nn.MSELoss(reduction='mean')
    l1_loss = nn.L1Loss(reduction='mean')
    cfg.lambda1 = 0.01 if cfg.model in ('unet') and cfg.input_type == 'iq' else cfg.lambda1
    criterion = lambda x, y: mse_loss(x, y) + l1_loss(x, torch.zeros_like(y)) * cfg.lambda1
    train_step = 0
    val_step = 0

    # Gaussian with gradually decreasing sigma
    exp = 2
    sigmas = torch.linspace(3.5**(-exp), 1, epochs)**(-1/exp)
    g_len = 7+cfg.upscale_factor//2*2
    psf_heatmap = torch.from_numpy(matlab_style_gauss2D(shape=(g_len,g_len), sigma=float(sigmas[-1])))
    gfilter = torch.reshape(psf_heatmap, [1, 1, g_len, g_len]).to(cfg.device)
    if cfg.model.__contains__('mspcn') and cfg.input_type == 'iq':
        cfg.lambda0 = 50
    elif cfg.model in ('unet') and cfg.input_type == 'iq':
        cfg.lambda0 = 1

    # transformation
    t_mats = get_inverse_mapping(dataset, p=6, weights_opt=False, point_num=1e4) if cfg.input_type == 'rf' else [[],[],[]]

    # training
    for epoch in range(1, epochs+1):
        # Gaussian with gradually decreasing sigma
        psf_heatmap = torch.from_numpy(matlab_style_gauss2D(shape=(g_len,g_len), sigma=float(sigmas[epoch-1])))
        gfilter = torch.reshape(psf_heatmap, [1, 1, g_len, g_len]).to(cfg.device)
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs, true_masks = batch[:2] if cfg.input_type == 'iq' else (batch[0].flatten(0, 1), batch[1].flatten(0, 1))

                # skip blank frames (avoid learning from false frames)
                if torch.any(imgs.view(imgs.shape[0], -1).sum() == 0) and torch.any(true_masks.view(true_masks.shape[0], -1).sum() > 0):
                    continue

                imgs = imgs.to(device=cfg.device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=cfg.device, dtype=torch.long)

                with torch.autocast(cfg.device if cfg.device != 'mps' else 'cpu', enabled=amp):

                    predictions = model(imgs)

                    # mask blurring
                    blur_masks = F.conv2d(true_masks.float(), gfilter, padding=gfilter.shape[-1]//2)
                    blur_masks /= blur_masks.max()
                    blur_masks *= cfg.lambda0
                    if cfg.model == 'mspcn' and cfg.input_type == 'iq':
                        predictions = F.conv2d(predictions, gfilter, padding=gfilter.shape[-1]//2)
                        
                    loss = criterion(predictions.squeeze(1), blur_masks.squeeze(1).float())

                optimizer.zero_grad(set_to_none=True)
                scale = grad_scaler.get_scale()
                grad_scaler.update()
                skip_lr_schedule = scale > grad_scaler.get_scale()

                if not skip_lr_schedule:
                    grad_scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    grad_scaler.step(optimizer)

                if cfg.logging:
                    wb.log({
                        'train loss': loss.item(),
                        'train_step': train_step,
                    })
                train_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(imgs.shape[0])

                # evaluation
                if train_step % division_step == 0 and division_step > 0 and not skip_lr_schedule:
                    val_step = evaluate(model, val_loader, epoch, val_step, criterion, amp, cfg, wb, t_mats, gfilter)
                    histograms = {}
                    for tag, value in model.named_parameters():
                        tag = tag.replace('/', '.')
                        if not torch.isinf(value).any() and not torch.isnan(value).any():
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        if not torch.isinf(value.grad).any() and not torch.isnan(value.grad).any():
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                    if cfg.logging:
                        wb.log({
                            **histograms,
                            'lr': optimizer.param_groups[0]['lr'],
                            'epoch': epoch,
                        })
    
        scheduler.step()

    # save weights
    if cfg.logging:
        dir_checkpoint = Path('./ckpts/')
        dir_checkpoint.mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        torch.save(state_dict, str(dir_checkpoint / (wb.name+str('_ckpt_epoch{}.pth'.format(epoch)))))
        logging.info(f'Checkpoint {epoch} saved!')

    # ideal threshold estimation from full frame examples
    val_loader.transforms = [ArgsToTensor(), NormalizeImage()] if cfg.input_type == 'iq' else [ArgsToTensor(), NormalizeVol()]
    threshold_list = []
    for batch in val_loader:

        # move images and labels to correct device and type
        imgs, true_masks = batch[:2] if cfg.input_type == 'iq' else (batch[0].flatten(0,1), batch[1].flatten(0,1))
        imgs = imgs.to(device=cfg.device, dtype=torch.float32)
        true_masks = (true_masks>0).to(device=cfg.device, dtype=torch.bool)

        # predict the mask
        predictions = model(imgs).detach()
        predictions = non_max_supp_torch(predictions, cfg.nms_size)

        if true_masks.sum() > 0 and torch.any(~torch.isnan(predictions)):
            roc_threshold = estimate_threshold(true_masks.squeeze(), predictions.squeeze())
            threshold_list.append(roc_threshold)

    roc_threshold = np.mean([el for el in threshold_list if el != float('Inf') and el != float('NaN')])
    print('mean_ROC_threshold: %s' % float(roc_threshold))

    if cfg.logging:
        wb.log({'mean_ROC_threshold': roc_threshold})
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
        model = SlounAdaptUNet(n_channels=in_channels, n_classes=1)
    elif cfg.model == 'mspcn':
        model = MSPCN(upscale_factor=cfg.upscale_factor, in_channels=in_channels)
    elif cfg.model == 'sgspcn':
        model = MSPCN(upscale_factor=cfg.upscale_factor, in_channels=in_channels, semi_global_scale=16)
    else:
        raise Exception('Model name not recognized')

    model = model.to(memory_format=torch.channels_last)
    model.to(device=cfg.device)

    if cfg.fine_tune:
        ckpt_paths = [fn for fn in Path('./ckpts').iterdir() if fn.name.startswith(cfg.model_file.split('_')[0])]
        state_dict = torch.load(str(ckpt_paths[0]), map_location=cfg.device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {cfg.model_file}')

    train_model(
        model=model,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        val_percent=0.1,
        amp=False,
        cfg = cfg
    )
