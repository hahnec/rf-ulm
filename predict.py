import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchinfo import summary
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
import wandb
import time
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.cluster import DBSCAN
from skimage.transform import rescale
from skimage.metrics import structural_similarity as ssim
from simple_tracker.tracks2img import tracks2img

from datasets.pala_dataset.pala_iq import PalaDatasetIq
from datasets.pala_dataset.pala_rf import PalaDatasetRf
from datasets.pala_dataset.utils.pala_error import rmse_unique
from datasets.pala_dataset.utils.radial_pala import radial_pala
from datasets.pala_dataset.utils.centroids import regional_mask
from models.unet import UNet, SlounUNet, SlounAdaptUNet
from models.mspcn import MSPCN
from models.edsr import EDSR
from utils.nms_funs import non_max_supp, non_max_supp_torch
from utils.point_align import align_points, get_pala_error
from utils.samples_points_map import get_inverse_mapping
from utils.srgb_conv import srgb_conv
from utils.utils import plot_img_and_mask
from utils.transform import Normalize, NormalizeVol
from utils.point_fusion import cluster_points
from utils.dithering import dithering


if __name__ == '__main__':

    # load configuration
    cfg = OmegaConf.load('./config.yml')

    # override loaded configuration with CLI arguments
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

    if cfg.logging:
        wb = wandb.init(project='SR-ULM-INFER', resume='allow', anonymous='must', config=cfg, group=cfg.logging)
        wb.config.update(cfg)

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
    else:
        raise Exception('Model name not recognized')

    logging.info(f'Loading model {cfg.model_file}')
    logging.info(f'Using device {cfg.device}')

    model.to(device=cfg.device)
    ckpt_paths = [fn for fn in Path('./ckpts').iterdir() if fn.name.startswith(cfg.model_file.split('_')[0])]
    state_dict = torch.load(str(ckpt_paths[0]), map_location=cfg.device)
    model.load_state_dict(state_dict)
    model.eval()

    if cfg.input_type == 'iq':
        DatasetClass = PalaDatasetIq
        transforms = [Normalize(mean=0, std=1)]
        from datasets.pala_dataset.utils.collate_fn_iq import collate_fn
    elif cfg.input_type == 'rf':
        DatasetClass = PalaDatasetRf
        transforms = [NormalizeVol()]
        from datasets.pala_dataset.utils.collate_fn_rf import collate_fn
        cluster_obj = DBSCAN(eps=cfg.eps, min_samples=1) if cfg.eps > 0 else None
    dataset = DatasetClass(
        dataset_path=cfg.data_dir,
        transforms=transforms,
        sequences = list(range(1, 101)) if cfg.data_dir.lower().__contains__('rat') else cfg.sequences,
        rescale_factor = cfg.rescale_factor,
        upscale_factor = cfg.upscale_factor,
        transducer_interp = True,
        tile_opt = False,
        scale_opt = cfg.model.lower().__contains__('unet'),
        clutter_db = cfg.clutter_db,
        temporal_filter_opt = cfg.data_dir.lower().__contains__('rat'),
        compound_opt = cfg.data_dir.lower().__contains__('rat'),
        pow_law_opt = cfg.pow_law_opt,
        )

    # data-related configuration
    cfg.wavelength = float(dataset.get_key('wavelength'))
    cfg.origin_x = float(dataset.get_key('Origin')[0])
    cfg.origin_z = float(dataset.get_key('Origin')[2])
    origin = np.array([cfg.origin_x, cfg.origin_z])
    cfg.wv_idcs = [0] if cfg.input_type == 'iq' else cfg.wv_idcs

    # transformation
    t_mats = get_inverse_mapping(dataset, p=6, weights_opt=False, point_num=1e4) if cfg.input_type == 'rf' else np.stack([np.eye(3), np.eye(3), np.eye(3)])
    
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

            tic = time.process_time()

            wv_es_points = []
            for wv_idx in cfg.wv_idcs:
                img, true_mask, gt_pts = batch[:3] if cfg.input_type == 'iq' else (batch[0][:, wv_idx], batch[1][:, wv_idx], batch[4])

                img = img.to(device=cfg.device, dtype=torch.float32)

                with torch.no_grad():
                    infer_start = time.process_time()
                    output = model(img)
                    infer_time = time.process_time() - infer_start

                # non-maximum suppression
                nms_start = time.process_time()
                if cfg.nms_size is not None:
                    mask = non_max_supp_torch(output, cfg.nms_size)
                    mask = mask > cfg.nms_threshold
                    mask = mask.squeeze(1)
                else:
                    # cpu-based local maxima (time-consuming for large frames)
                    mask = regional_mask(output.squeeze().cpu().numpy(), th=cfg.nms_threshold)
                    mask = torch.tensor(mask, device=cfg.device)[None, ...]
                nms_time = time.process_time() - nms_start

                pts_start = time.process_time()
                es_points, gt_points = align_points(mask, gt_pts, t_mat=t_mats[wv_idx], cfg=cfg, sr_img=output)
                pts_time = time.process_time() - pts_start
                
                wv_es_points.append(es_points)

            if len(cfg.wv_idcs) > 1:
                pts = np.hstack([wv_es_points[1][0], wv_es_points[0][0], wv_es_points[2][0]])
                # fuse points using DBSCAN when eps > 0 and 
                es_points = [cluster_points(pts.T, cluster_obj=cluster_obj).T] if pts.size > 0 and cfg.eps > 0 else [pts]
            else:
                es_points = wv_es_points[0]

            all_pts.append(es_points[0].T)
            all_pts_gt.append(gt_points[0].T)

            frame_time = time.process_time() - tic
            
            if False:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(*gt_points[0], 'rx')
                plt.plot(*es_points[0], 'b+')
                plt.show()

            # localization assessment
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
                    'FrameTime': frame_time,
                    'InferTime': infer_time,
                    'NMS_Time': nms_time,
                    'PointsTime': pts_time,
                    'frame': int(i),
                })
            
            pbar.update(i)

    errs = torch.tensor(ac_rmse_err)
    sres_rmse_mean = torch.nanmean(errs[..., 0], axis=0)
    sres_rmse_std = torch.std(errs[..., 0][~torch.isnan(errs[..., 0])], axis=0)
    print('Acc. Errors: %s' % str(torch.nanmean(errs, axis=0)))

    # remove empty arrays
    all_pts = [p for p in all_pts if p.size > 0]
    all_pts_gt = [p for p in all_pts_gt if p.size > 0]

    # final resolution handling
    img_size = np.array([84, 134])
    gtru_ulm_img, _ = tracks2img(all_pts_gt, img_size=img_size, scale=10, mode='all_in')
    img_shape = np.array(img.shape[-2:])[::-1] if cfg.input_type == 'rf' else img_size
    if cfg.dither:
        # dithering
        y_factor, x_factor = img_shape / img_size
        all_pts = dithering(all_pts, 10, cfg.upscale_factor, x_factor, y_factor)

    if cfg.upscale_factor < 10 and not cfg.dither:
        sres_ulm_img, _ = tracks2img(all_pts, img_size=img_size, scale=cfg.upscale_factor, mode='tracks' if cfg.track else 'all_in', fps=dataset.frames_per_seq)
        # upscale input frame
        if cfg.upscale_factor != 1:
            import cv2
            sres_ulm_img = cv2.resize(sres_ulm_img, 10*img_size[::-1], interpolation=cv2.INTER_CUBIC)
            sres_ulm_img[sres_ulm_img<0] = 0
    else:
        sres_ulm_img, _ = tracks2img(all_pts, img_size=img_size, scale=10, mode='tracks' if cfg.track else 'all_in', fps=dataset.frames_per_seq)

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
    sres_ulm_map = img_color_map(img=normalize(sres_ulm_img))
    gtru_ulm_map = img_color_map(img=normalize(gtru_ulm_img))

    if cfg.logging:
        wandb.log({"sres_ulm_img": wandb.Image(sres_ulm_map)})
        wandb.log({"gtru_ulm_img": wandb.Image(gtru_ulm_map)})
        wandb.summary['TotalRMSE'] = sres_rmse_mean
        wandb.summary['TotalRMSEstd'] = sres_rmse_std
        wandb.summary['TotalJaccard'] = torch.nanmean(errs[..., 3], axis=0)
        wandb.summary['SSIM'] = ssim(gtru_ulm_img, sres_ulm_img, data_range=sres_ulm_img.max()-sres_ulm_img.min())
        wandb.summary['TotalParameters'] = int(str(summary(model)).split('\n')[-3].split(' ')[-1].replace(',',''))
        wandb.save(str(Path('.') / 'logged_errors.csv'))
        wandb.finish()
