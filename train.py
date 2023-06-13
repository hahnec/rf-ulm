import argparse
import logging
import os
import random
import sys
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from omegaconf import OmegaConf

from evaluate import evaluate, non_max_supp
from unet import UNet, SlounUNet, SlounAdaptUNet
from mspcn.model import Net
from mspcn.main import matlab_style_gauss2D
from utils.dataset_pala import InSilicoDataset
from utils.dice_score import dice_loss
from utils.transform import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, GaussianNoise

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')


img_norm = lambda x: (x-x.min())/(x.max()-x.min()) if (x.max()-x.min()) != 0 else x

transforms = [RandomHorizontalFlip(), RandomVerticalFlip(), RandomRotation(degree=5), GaussianNoise()]


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        cfg = None,
):
    # 1. Create dataset
    dataset = InSilicoDataset(
        dataset_path=cfg.data_dir,
        transforms = transforms,
        rf_opt = False,
        sequences = [15, 16, 17, 18, 19],
        rescale_factor = 8 if cfg.model.__contains__('unet') else 4,
        rescale_frame = True if cfg.model.__contains__('unet') else False,
        blur_opt=cfg.blur_opt if cfg.model.__contains__('unet') else False,
        tile_opt=True if cfg.model.__contains__('unet') else False,
        )

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    if cfg.logging:
        experiment = wandb.init(project='U-Net', resume='allow', anonymous='must', config=cfg)
        experiment.config.update(
            dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
        )

        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Images scaling:  {img_scale}
            Mixed Precision: {amp}
        ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    #optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.MSELoss(reduction='mean')
    l1loss = nn.L1Loss(reduction='mean')
    global_step = 0
    
    # mSPCN Gaussian
    psf_heatmap = torch.from_numpy(matlab_style_gauss2D(shape=(7,7),sigma=1))
    gfilter = torch.reshape(psf_heatmap, [1, 1, 7, 7])
    if cfg.device == 'cuda': gfilter.cuda()

    # 5. Begin training
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, masks_true = batch[:2]

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                masks_true = masks_true.to(device=device)#, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)

                    if cfg.model == 'mspcn':
                        masks_pred = F.conv2d(masks_pred, gfilter)
                        masks_true = F.conv2d(masks_true, gfilter)
                        
                    loss = criterion(masks_pred.squeeze(1), masks_true.squeeze(1).float())
                    loss += l1loss(masks_pred.squeeze(1), torch.zeros_like(masks_pred.squeeze(1))) * 0.01


                # activation followed by non-maximum suppression
                #masks_pred = torch.sigmoid(masks_pred)
                #imgs_nms = non_max_supp(masks_pred)
                #masks_nms = imgs_nms > cfg.nms_threshold

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                if cfg.logging:
                    experiment.log({
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not torch.isinf(value).any() and not torch.isnan(value).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not torch.isinf(value.grad).any() and not torch.isnan(value.grad).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                        
                        val_score, pala_err_batch, masks_nms, threshold = evaluate(model, val_loader, device, amp, cfg)
                        scheduler.step(val_score)

                        rmse, precision, recall, jaccard, tp_num, fp_num, fn_num = pala_err_batch

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            if cfg.logging:
                                experiment.log({
                                    'learning rate': optimizer.param_groups[0]['lr'],
                                    'validation Dice': val_score,
                                    'images': wandb.Image(images[0].cpu()),
                                    'masks': {
                                        'true': wandb.Image(img_norm(masks_true[0].float().cpu())*255),
                                        'pred': wandb.Image(img_norm(masks_pred[0].float().cpu())*255),    #(masks_pred.argmax(dim=1)[0]).float().cpu()),#
                                        'nms': wandb.Image(img_norm(masks_nms[0].float().cpu())*255),
                                    },
                                    'step': global_step,
                                    'epoch': epoch,
                                    'rmse': rmse,
                                    'precision': precision,
                                    'recall': recall,
                                    'jaccard': jaccard,
                                    'threshold': threshold,
                                    'avg_detected': float(masks_nms[0].float().cpu().sum()),
                                    'pred_max': float(masks_pred[0].float().cpu().max()),
                                    **histograms
                                })
                        except Exception as e:
                            print('Validation upload failed')
                            print(e)

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            #state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    cfg = OmegaConf.load('./pala_unet.yml')

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Model selection
    if cfg.model == 'unet':
        # UNet model
        # n_channels=3 for RGB images
        # n_classes is the number of probabilities you want to get per pixel
        model = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear)
        #model = SlounUNet(n_channels=1, n_classes=1, bilinear=False)
        model = SlounAdaptUNet(n_channels=1, n_classes=1, bilinear=False)
    elif cfg.model == 'mspcn':
        # mSPCN model
        model = Net(upscale_factor=4)
    else:
        raise Exception('Model name not recognized')

    model = model.to(memory_format=torch.channels_last)

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            cfg = cfg
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
