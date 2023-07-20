import argparse
import logging
import os
import time

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
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from skimage.transform import rescale
from skimage.metrics import structural_similarity
from simple_tracker.tracks2img import tracks2img

from datasets.pala_dataset.pala_iq import PalaDatasetIq
from datasets.pala_dataset.pala_rf import PalaDatasetRf
from datasets.pala_dataset.utils.pala_error import rmse_unique
from datasets.pala_dataset.utils.radial_pala import radial_pala
from datasets.pala_dataset.utils.centroids import regional_mask
from unet import UNet, SlounUNet, SlounAdaptUNet
from mspcn.model import Net
from evaluate import non_max_supp, get_pala_error
from utils.srgb_conv import srgb_conv
from utils.utils import plot_img_and_mask
from utils.transform import NormalizeVol


def predict_img(net,
                img,
                device,
                scale_factor=1,
                out_threshold=0.5
                ):
    net.eval()
    #img = torch.from_numpy(BasicDataset.preprocess(None, img, scale_factor, is_mask=False))
    #img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        start = time.time()
        output = net(img)
        comp_time = time.time() - start
        #output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        output = output.squeeze().cpu()
        # non-maximum suppression
        if False:
            nms = non_max_supp(output)
            mask = nms > out_threshold
            mask = mask.long().numpy()
        else:
            mask = regional_mask(output.numpy(), th=out_threshold)

    return mask[None, ...], output, comp_time


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


def img_color_map(img=None, cmap='inferno'):

    # get color map
    colormap = plt.get_cmap(cmap)

    # apply color map omitting alpha channel
    img = colormap(img)[..., :3]

    return img


if __name__ == '__main__':
    args = get_args()
    #logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    #in_files = args.input
    #out_files = get_output_filenames(args)

    cfg = OmegaConf.load('./pala_unet.yml')

    if cfg.logging:
        experiment = wandb.init(project='pulm', resume='allow', anonymous='must', config=cfg)
        experiment.config.update(cfg)

    # Model selection
    if cfg.model == 'unet':
        # UNet model
        # n_channels=3 for RGB images
        # n_classes is the number of probabilities you want to get per pixel
        net = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear)
        #model = SlounUNet(n_channels=1, n_classes=1, bilinear=False)
        net = SlounAdaptUNet(n_channels=1, n_classes=1, bilinear=False)
    elif cfg.model == 'mspcn':
        # mSPCN model
        net = Net(upscale_factor=cfg.upscale_factor)
    else:
        raise Exception('Model name not recognized')

    device = torch.device(cfg.device)
    logging.info(f'Loading model {cfg.model_path}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(cfg.model_path, map_location=device)
    mask_values = state_dict.pop('mask_values') if 'mask_values' in state_dict.keys() else None
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    if cfg.input_type == 'iq':
        DatasetClass = PalaDatasetIq
        transforms = []
        collate_fn = None
    elif cfg.input_type == 'rf':
        DatasetClass = PalaDatasetRf
        transforms = [NormalizeVol()]
        from datasets.pala_dataset.utils.collate_fn import collate_fn
    dataset = DatasetClass(
        dataset_path=cfg.data_dir,
        rf_opt = False,
        sequences = list(range(1, 16)),
        rescale_factor = cfg.rescale_factor,
        upscale_factor = cfg.upscale_factor,
        tile_opt = True if cfg.model.__contains__('unet') else False,
        clutter_db = cfg.clutter_db,
        temporal_filter_opt = False,
        transforms=transforms,
        )

    wavelength = 9.856e-05
    origin = np.array([-72,  16])

    t_mat = np.loadtxt('./t_mat.txt')

    num_workers = min(4, os.cpu_count())
    loader_args = dict(batch_size=1, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(dataset, shuffle=False, drop_last=True, **loader_args)
    ac_rmse_err = []
    all_pts = []
    all_pts_gt = []
    tic = time.process_time()
    for i, batch in enumerate(test_loader):
        with tqdm(total=len(test_loader), desc=f'Frame {i}/{len(test_loader)}', unit='img') as pbar:
            #logging.info(f'Predicting image {filename} ...')
            #img = Image.open(filename)
            img, true_mask, gt_pts = batch[:3] if cfg.input_type == 'iq' else (batch[2][:, 1].unsqueeze(1), batch[-2][:, 1].unsqueeze(1), batch[1])

            mask, output, comp_time = predict_img(net=net,
                            img=img,
                            scale_factor=args.scale,
                            out_threshold=cfg.nms_threshold,
                            device=device)

            # points alignment
            gt_points = gt_pts[:, ~(torch.isnan(gt_pts.squeeze()).sum(-1) > 0)].numpy()[:, ::-1]
            gt_points = gt_points.swapaxes(-2, -1)
            gt_points = gt_points[:, ::-1, :]
            es_points = np.array(np.nonzero(mask))[::-1, :] #- .5
            if cfg.input_type == 'rf':
                es_points[2] = 1
                es_points[:2, :] = es_points[:2, :][::-1, :] / cfg.upscale_factor
                es_points = t_mat @ es_points
                es_points[:2, :] = es_points[:2, :][::-1, :]
            es_points = es_points[:2, ...][None, ...]

            es_points /= wavelength
            gt_points /= wavelength

            pts_es = (es_points[0] + origin[:, None]).T
            pts_gt = (gt_points[0] + origin[:, None]).T
            all_pts.append(pts_es)
            all_pts_gt.append(pts_gt)

            iter_time = time.process_time() - tic
            
            if False:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(*gt_points[0], 'rx')
                plt.plot(*es_points[0], 'b+')
                plt.show()

            # localization assessment
            output = output.float().squeeze().cpu().numpy()
            result = get_pala_error(es_points, gt_points, upscale_factor=cfg.upscale_factor, sr_img=output, avg_weight_opt=cfg.avg_weight_opt, radial_sym_opt=cfg.radial_sym_opt)
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
                    'U-Net/InferTime': comp_time,
                    'U-Net/FrameTime': iter_time,
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

            if False:
                # calculate the g-mean for each threshold
                fpr, tpr, thresholds = roc_curve(true_mask.float().numpy().flatten(), output.flatten())
                #precision, recall, thresholds = precision_recall_curve(true_mask.float().numpy().flatten(), output.float().numpy().flatten())
                gmeans = (tpr * (1-fpr))**.5
                th_idx = np.argmax(gmeans)
                threshold = thresholds[th_idx]

            if gt_pts.size == 0:
                continue

            tic = time.process_time()

errs = torch.tensor(ac_rmse_err)
unet_rmse_mean = torch.nanmean(errs[..., 0], axis=0)
unet_rmse_std = torch.std(errs[..., 0][~torch.isnan(errs[..., 0])], axis=0)
print('Acc. Errors: %s' % str(torch.nanmean(errs, axis=0)))

# remove empty arrays
all_pts = [p for p in all_pts if p.size > 0]
all_pts_gt = [p for p in all_pts_gt if p.size > 0]

unet_ulm_img, _ = tracks2img((np.vstack(all_pts)-origin), img_size=np.array([84, 134]), scale=cfg.upscale_factor, mode='all_in')
gtru_ulm_img, _ = tracks2img((np.vstack(all_pts_gt)-origin)[:, ::-1], img_size=np.array([84, 134]), scale=10, mode='all_in')

scale_factor = 10 / cfg.upscale_factor
unet_ulm_img = rescale(unet_ulm_img, scale_factor, mode='reflect', anti_aliasing=True) if scale_factor != 1 else unet_ulm_img

normalize = lambda x: (x-x.min())/(x.max()-x.min()) if x.max()-x.min() > 0 else x-x.min()

# gamma
unet_ulm_img **= cfg.gamma
gtru_ulm_img **= cfg.gamma

# sRGB gamma correction
unet_ulm_img = srgb_conv(normalize(unet_ulm_img))
gtru_ulm_img = srgb_conv(normalize(gtru_ulm_img))

# color mapping
unet_ulm_img = img_color_map(img=normalize(unet_ulm_img), cmap='inferno')
gtru_ulm_img = img_color_map(img=normalize(gtru_ulm_img), cmap='inferno')

if cfg.logging:
    wandb.summary['ULM/TotalRMSE'] = unet_rmse_mean
    wandb.summary['ULM/TotalRMSEstd'] = unet_rmse_std
    wandb.summary['ULM/TotalJaccard'] = torch.nanmean(errs[..., 3], axis=0)
    wandb.summary['ULM/SSIM'] = structural_similarity(gtru_ulm_img, unet_ulm_img, channel_axis=2)
    wandb.log({"unet_ulm_img": wandb.Image(unet_ulm_img)})
    wandb.log({"gtru_ulm_img": wandb.Image(gtru_ulm_img)})
    wandb.save(str(Path('.') / 'logged_errors.csv'))