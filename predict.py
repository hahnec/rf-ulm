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
from evaluate import non_max_supp, non_max_supp_torch, get_pala_error
from evaluate import align_points
from utils.srgb_conv import srgb_conv
from utils.utils import plot_img_and_mask
from utils.transform import NormalizeVol


if __name__ == '__main__':

    # load configuration
    cfg = OmegaConf.load('./config.yml')

    # override loaded configuration with CLI arguments
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

    if cfg.logging:
        wb = wandb.init(project='SR-ULM-INFER', resume='allow', anonymous='must', config=cfg)
        wb.config.update(cfg)

    # Model selection
    in_channels = 1
    if cfg.model == 'unet':
        # UNet model
        net = SlounAdaptUNet(n_channels=in_channels, n_classes=1, bilinear=False)
    elif cfg.model == 'mspcn':
        # mSPCN model
        net = Net(upscale_factor=cfg.upscale_factor, in_channels=in_channels)
    else:
        raise Exception('Model name not recognized')

    device = torch.device(cfg.device)
    logging.info(f'Loading model {cfg.model_path}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(Path('./checkpoints') / cfg.model_path, map_location=device)
    mask_values = state_dict.pop('mask_values') if 'mask_values' in state_dict.keys() else None
    net.load_state_dict(state_dict)

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
        transforms=transforms,
        sequences = list(range(1, 121)) if cfg.data_dir.lower().__contains__('rat') else cfg.sequences,
        rescale_factor = cfg.rescale_factor,
        upscale_factor = cfg.upscale_factor,
        tile_opt = cfg.model.lower().__contains__('unet'),
        clutter_db = cfg.clutter_db,
        temporal_filter_opt = cfg.data_dir.lower().__contains__('rat'),
        )

    # data-related configuration
    cfg.wavelength = float(dataset.get_key('wavelength'))
    cfg.origin_x = float(dataset.get_key('Origin')[0])
    cfg.origin_z = float(dataset.get_key('Origin')[2])
    origin = np.array([cfg.origin_x, cfg.origin_z])
    wv_idx = 1
    name_ext = '_' + str(int(cfg.upscale_factor)) + '_' + str(int(cfg.rescale_factor))
    t_mats = torch.tensor(np.load('./t_mats' + name_ext + '.npy')).to(cfg.device)
    
    # data loader
    num_workers = min(4, os.cpu_count())
    loader_args = dict(batch_size=1, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(dataset, shuffle=False, drop_last=True, **loader_args)

    ac_rmse_err = []
    all_pts = []
    all_pts_gt = []
    tic = time.process_time()
    for i, batch in enumerate(test_loader):
        with tqdm(total=len(test_loader), desc=f'Frame {i}/{len(test_loader)}', unit='img') as pbar:

            img, true_mask, gt_pts = batch[:3] if cfg.input_type == 'iq' else (batch[2][:, wv_idx].unsqueeze(1), batch[-2][:, wv_idx].unsqueeze(1), batch[1])

            net.eval()
            img = img.to(device=cfg.device, dtype=torch.float32)

            with torch.no_grad():
                infer_start = time.process_time()
                output = net(img)
                infer_time = time.process_time() - infer_start
                output = output.squeeze().cpu()

            # non-maximum suppression
            if False:
                nms = non_max_supp_torch(output, cfg.nms_size)
                mask = nms > cfg.nms_threshold
                mask = mask.long().numpy()
            else:
                mask = regional_mask(output.numpy(), th=cfg.nms_threshold)

            masks = mask[None, ...]

            es_points, gt_points = align_points(torch.tensor(masks, device=cfg.device), gt_pts, t_mat=t_mats[wv_idx], cfg=cfg, sr_img=np.array(output))

            pts_es = (es_points[0] + origin[:, None]).T
            pts_gt = (gt_points[0] + origin[:, None]).T
            all_pts.append(pts_es)
            all_pts_gt.append(pts_gt)

            frame_time = time.process_time() - tic
            
            if False:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(*gt_points[0], 'rx')
                plt.plot(*es_points[0], 'b+')
                plt.show()

            # localization assessment
            output = output.float().squeeze().cpu().numpy()
            result = get_pala_error(es_points, gt_points)[0]
            ac_rmse_err.append(result)

            if cfg.logging:
                wandb.log({
                    'RMSE': result[0],
                    'Precision': result[1],
                    'Recall': result[2],
                    'Jaccard': result[3],
                    'TruePositive': result[4],
                    'FalsePositive': result[5],
                    'FalseNegative': result[6],
                    'InferTime': infer_time,
                    'FrameTime': frame_time,
                    'frame': int(i),
                })

            if False:
                # calculate the g-mean for each threshold
                fpr, tpr, thresholds = roc_curve(true_mask[0].float().numpy().flatten(), output[0].flatten())
                #precision, recall, thresholds = precision_recall_curve(true_mask.float().numpy().flatten(), output.float().numpy().flatten())
                gmeans = (tpr * (1-fpr))**.5
                th_idx = np.argmax(gmeans)
                threshold = thresholds[th_idx]

            tic = time.process_time()

errs = torch.tensor(ac_rmse_err)
sres_rmse_mean = torch.nanmean(errs[..., 0], axis=0)
sres_rmse_std = torch.std(errs[..., 0][~torch.isnan(errs[..., 0])], axis=0)
print('Acc. Errors: %s' % str(torch.nanmean(errs, axis=0)))

# remove empty arrays
all_pts = [p for p in all_pts if p.size > 0]
all_pts_gt = [p for p in all_pts_gt if p.size > 0]

sres_ulm_img, _ = tracks2img((np.vstack(all_pts)-origin)[:, ::-1]-origin, img_size=np.array([84, 134]), scale=10, mode='all_in')
gtru_ulm_img, _ = tracks2img((np.vstack(all_pts_gt)-origin)[:, ::-1]-origin, img_size=np.array([84, 134]), scale=10, mode='all_in')

# gamma
sres_ulm_img **= cfg.gamma
gtru_ulm_img **= cfg.gamma

# sRGB gamma correction
normalize = lambda x: (x-x.min())/(x.max()-x.min()) if x.max()-x.min() > 0 else x-x.min()
sres_ulm_img = srgb_conv(normalize(sres_ulm_img))
gtru_ulm_img = srgb_conv(normalize(gtru_ulm_img))

# color mapping
cmap = 'hot' if str(cfg.data_dir).lower().__contains__('rat') else 'inferno'
img_color_map = lambda img, cmap=cmap: plt.get_cmap(cmap)(img)[..., :3]
sres_ulm_img = img_color_map(img=normalize(sres_ulm_img))
gtru_ulm_img = img_color_map(img=normalize(gtru_ulm_img))

if cfg.logging:
    wandb.summary['TotalRMSE'] = sres_rmse_mean
    wandb.summary['TotalRMSEstd'] = sres_rmse_std
    wandb.summary['TotalJaccard'] = torch.nanmean(errs[..., 3], axis=0)
    wandb.summary['SSIM'] = structural_similarity(gtru_ulm_img, sres_ulm_img, channel_axis=2)
    wandb.log({"sres_ulm_img": wandb.Image(sres_ulm_img)})
    wandb.log({"gtru_ulm_img": wandb.Image(gtru_ulm_img)})
    wandb.save(str(Path('.') / 'logged_errors.csv'))
