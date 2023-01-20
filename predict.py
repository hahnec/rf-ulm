import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from omegaconf import OmegaConf
import wandb
import time

from evaluate import non_max_supp, get_pala_error
from utils.data_loading import BasicDataset
from unet import UNet, SlounUNet, SlounAdaptUNet
from utils.dataset_pala import InSilicoDataset
from utils.utils import plot_img_and_mask
from utils.pala_error import rmse_unique
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve


def predict_img(net,
                img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    #img = torch.from_numpy(BasicDataset.preprocess(None, img, scale_factor, is_mask=False))
    #img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        start = time.time()
        output = net(img)
        comp_time = time.time() - start
        #output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        output = output.cpu()
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            # non-maximum suppression
            nms = non_max_supp(output)
            mask = nms > cfg.nms_threshold

    return mask[0].long().numpy(), output, comp_time


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=False)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    #logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    #in_files = args.input
    #out_files = get_output_filenames(args)

    cfg = OmegaConf.load('./pala_unet.yml')

    if cfg.logging:
        experiment = wandb.init(project='pulm', resume='allow', anonymous='must', config=cfg)
        experiment.config.update(cfg)

    net = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear)
    #net = SlounUNet(n_channels=1, n_classes=1, bilinear=False)
    net = SlounAdaptUNet(n_channels=1, n_classes=1, bilinear=False)

    device = torch.device(cfg.device)
    logging.info(f'Loading model {cfg.model_path}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(cfg.model_path, map_location=device)
    mask_values = state_dict.pop('mask_values') if 'mask_values' in state_dict.keys() else None
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    dataset = InSilicoDataset(
        dataset_path=cfg.data_dir,
        rf_opt = False,
        sequences = list(range(0, 15)),
        rescale_factor = cfg.rescale_factor,
        blur_opt = False,
        tile_opt = False,
        )
    wavelength = 9.856e-05
    origin = np.array([-72,  16])

    loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(dataset, shuffle=False, drop_last=True, **loader_args)
    ac_rmse_err = []
    for i, batch in enumerate(test_loader):
        with tqdm(total=len(dataset), desc=f'Frame {i}/{len(dataset)}', unit='img') as pbar:
            #logging.info(f'Predicting image {filename} ...')
            #img = Image.open(filename)
            img, true_mask, gt_pts = batch[:3]

            mask, output, comp_time = predict_img(net=net,
                            img=img,
                            scale_factor=args.scale,
                            out_threshold=cfg.nms_threshold,
                            device=device)

            gt_pts = gt_pts[:, ~(torch.isnan(gt_pts.squeeze()).sum(-1) > 0)].numpy()[:, ::-1]
            result = get_pala_error(mask, gt_pts, rescale_factor=cfg.rescale_factor)
            ac_rmse_err.append(result)

            if cfg.logging:
                wandb.log({
                    'U-Net/RMSE': result[0],
                    'U-Net/Precision': result[1],
                    'U-Net/Recall': result[2],
                    'U-Net/Jaccard': result[3],
                    'U-Net/TruePositive': result[4],
                    'U-Net/FalsePositive': result[5],
                    'U-Net/FalseNegative': result[6],
                    'U-Net/FrameTime': comp_time,
                    'frame': int(i),
                })

            if cfg.save_opt:
                out_filename = out_files[i]
                result = mask_to_image(mask, mask_values)
                result.save(out_filename)
                logging.info(f'Mask saved to {out_filename}')

            if cfg.plot_opt:
                logging.info(f'Visualizing results for image, close to continue...')
                plot_img_and_mask(img.squeeze(), mask.squeeze())

            fpr, tpr, thresholds = roc_curve(true_mask.float().numpy().flatten(), output.float().numpy().flatten())
            #precision, recall, thresholds = precision_recall_curve(true_mask.float().numpy().flatten(), output.float().numpy().flatten())

            # calculate the g-mean for each threshold
            gmeans = (tpr * (1-fpr))**.5
            th_idx = np.argmax(gmeans)
            threshold = thresholds[th_idx]

errs = torch.tensor(ac_rmse_err)
unet_rmse_mean = torch.nanmean(errs[..., 0], axis=0)
unet_rmse_std = torch.std(errs[..., 0][~torch.isnan(errs[..., 0])], axis=0)
print('Acc. Errors: %s' % str(errs))
if cfg.logging:
    wandb.summary['ULM/TotalRMSE'] = unet_rmse_mean
    wandb.summary['ULM/TotalRMSEstd'] = unet_rmse_std
    wandb.summary['ULM/TotalJaccard'] = torch.nanmean(errs[..., 3], axis=0)
    wandb.save(str(output_path / 'logged_errors.csv'))