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
from matplotlib.colors import LinearSegmentedColormap
from scipy import interpolate
import cv2
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.cluster import DBSCAN
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from simple_tracker.tracks2img import tracks2img

from datasets.pala_dataset.pala_iq import PalaDatasetIq
from datasets.pala_dataset.pala_rf import PalaDatasetRf
from datasets.pala_dataset.utils.pala_error import rmse_unique
from datasets.pala_dataset.utils.radial_pala import radial_pala
from datasets.pala_dataset.utils.centroids import regional_mask
from models.unet import UNet, SlounUNet, SlounAdaptUNet
from models.mspcn import MSPCN
from utils.nms_funs import non_max_supp_torch
from utils.point_align import align_points, get_pala_error
from utils.samples_points_map import get_inverse_mapping
from utils.srgb_conv import srgb_conv
from utils.utils import plot_img_and_mask
from utils.transform import ArgsToTensor, NormalizeImage, NormalizeVol
from utils.point_fusion import cluster_points
from utils.render_ulm import render_ulm_frame


normalize = lambda x: (x-x.min())/(x.max()-x.min()) if x.max()-x.min() > 0 else x-x.min()
img_color_map = lambda img, cmap: plt.get_cmap(cmap)(img)[..., :3]
truncate_outliers = lambda x, q=1e-4: np.where(x < np.quantile(x, q), np.quantile(x, q), np.where(x > np.quantile(x, 1-q), np.quantile(x, 1-q), x))
ulm_scale = lambda img, gamma: srgb_conv(normalize(truncate_outliers(img)**gamma))
ulm_align = lambda img, gamma, cmap: img_color_map(img=ulm_scale(img, gamma), cmap=cmap)
velo_cmap = LinearSegmentedColormap.from_list('custom_colormap', [(2/3, 1, 1),(0, 1/3, 1), (0, 0, 0), (1, 1/3, 0), (1, 1, 2/3)], N=2**8)


if __name__ == '__main__':

    # load configuration
    cfg = OmegaConf.load('./config.yml')

    # override loaded configuration with CLI arguments
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    cfg.invivo = cfg.data_dir.lower().__contains__('rat')

    # for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # transducer channels
    cfg.channel_num = 128 if not cfg.invivo or not hasattr(cfg, 'channel_num') or cfg.channel_num is None else cfg.channel_num

    if cfg.logging:
        wb = wandb.init(project='SR-ULM-INFER', resume='allow', anonymous='must', config=cfg, group=str(cfg.logging))
        wb.config.update(cfg)

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

    logging.info(f'Loading model {cfg.model_file}')
    logging.info(f'Using device {cfg.device}')

    model.to(device=cfg.device)
    ckpt_paths = [fn for fn in Path('./ckpts').iterdir() if fn.name.startswith(cfg.model_file.split('_')[0])]
    state_dict = torch.load(str(ckpt_paths[0]), map_location=cfg.device)
    model.load_state_dict(state_dict)
    model.eval()

    # initialize lists
    ac_rmse_err, all_pts, all_pts_gt, all_pts_indices, bmode_frames = [], [], [], [], []

    # dataset init
    if cfg.input_type == 'iq':
        DatasetClass = PalaDatasetIq
        transforms = [ArgsToTensor(), NormalizeImage()]
        from datasets.pala_dataset.utils.collate_fn_iq import collate_fn
    elif cfg.input_type == 'rf':
        DatasetClass = PalaDatasetRf
        transforms = [ArgsToTensor(), NormalizeVol()] if cfg.skip_bmode else [ArgsToTensor(), NormalizeImage()]
        from datasets.pala_dataset.utils.collate_fn_rf import collate_fn
        cluster_obj = DBSCAN(eps=cfg.eps, min_samples=1) if cfg.eps > 0 else None
    dataset = DatasetClass(
        dataset_path=cfg.data_dir,
        transforms=transforms,
        sequences = None,
        rescale_factor = cfg.rescale_factor,
        upscale_factor = cfg.upscale_factor,
        upscale_channels = cfg.channel_num,
        transducer_interp = True,
        scale_opt = cfg.model.lower().__contains__('unet'),
        bmode_depth_scale = 2 if cfg.invivo else 1,
        clutter_db = cfg.clutter_db,
        temporal_filter_opt = cfg.invivo,
        compound_opt = True,
        pow_law_opt = cfg.pow_law_opt,
        skip_bmode = cfg.skip_bmode,
        das_b4_temporal = cfg.das_b4_temporal,
        synth_gt = cfg.synth_gt,
    )

    # data-related configuration
    cfg.wavelength = float(dataset.get_key('wavelength'))
    cfg.origin_x = float(dataset.get_key('Origin')[0])
    cfg.origin_z = float(dataset.get_key('Origin')[2])
    cfg.wv_idcs = [0] if cfg.input_type == 'iq' or (cfg.input_type == 'rf' and not cfg.skip_bmode) else ([0,1,2,3,4] if str(cfg.data_dir).lower().__contains__('rat20') else cfg.wv_idcs)
    origin = np.array([cfg.origin_x, cfg.origin_z])
    img_size = np.array([84, 143]) if cfg.input_type == 'rf' else dataset.img_size
    cmap = 'hot' if str(cfg.data_dir).lower().__contains__('rat') else 'inferno'
    nms_size = cfg.upscale_factor if cfg.nms_size is None else cfg.nms_size
    h_acc = None

    # transformation
    t_mats = get_inverse_mapping(dataset, channel_num=cfg.channel_num, p=6, weights_opt=False, point_num=1e4) if cfg.input_type == 'rf' else np.stack([np.eye(3), np.eye(3), np.eye(3)])

    # iterate through sequences
    sequences = list(range(121)) if str(cfg.data_dir).lower().__contains__('rat') else cfg.sequences
    for sequence in sequences:

        # load next sequence
        dataset.read_sequence(sequence)

        # data loader
        loader_args = dict(batch_size=1, num_workers=0, pin_memory=False)
        test_loader = DataLoader(dataset, shuffle=False, **loader_args)

        for i, batch in enumerate(test_loader):
            with tqdm(total=len(test_loader), desc=f'Frame {i}/{len(test_loader)}', unit='img') as pbar:

                tic = time.process_time()

                imgs, true_masks, gt_pts = batch[:3] if cfg.input_type == 'iq' else (batch[0].squeeze(0), batch[1], batch[4])

                # use RF-based bmode frame
                if not cfg.skip_bmode and cfg.input_type == 'rf': imgs = batch[3]
                
                # move to desired device (GPU)
                imgs = imgs[cfg.wv_idcs].to(device=cfg.device, dtype=torch.float32)

                with torch.no_grad():
                    infer_start = time.process_time()
                    outputs = model(imgs)
                    infer_time = time.process_time() - infer_start

                # affine image warping
                if cfg.nms_size is None or cfg.save_image:
                    img = normalize(outputs.squeeze().cpu().permute(2,1,0).numpy())
                    new = np.zeros((84*cfg.upscale_factor, 143*cfg.upscale_factor, 3), dtype=float)
                    for ch in range(img.shape[-1]):
                        amat = t_mats[ch][:2, :3].copy()
                        amat[:2, -1] -= np.array([cfg.origin_x, cfg.origin_z]) 
                        new[..., ch] = cv2.warpAffine(img[..., ch], amat[:2, :3]*cfg.upscale_factor, (new.shape[1], new.shape[0]), flags=cv2.INTER_CUBIC)
                    u8_img = np.round(255*new).astype(np.uint8)

                    if cfg.save_image:
                        pil_img = Image.fromarray(u8_img)
                        frame_num = sequence*len(test_loader) + i
                        pil_img.save('./frames/'+str(frame_num).zfill(6)+".png")

                    # accumulate warped output frames
                    u8_img[u8_img<cfg.nms_threshold/outputs.cpu().numpy().max()*255] = 0
                    h_np = normalize(np.mean(u8_img, -1))
                    h_acc = h_acc + h_np/dataset.frames_per_seq if h_acc is not None else h_np/dataset.frames_per_seq

                # non-maximum suppression
                nms_start = time.process_time()
                if nms_size is not None:
                    masks = non_max_supp_torch(outputs, nms_size)
                    if cfg.nms_threshold is None:
                        # fix number of descending maximum values
                        point_num = 40
                        wave_idx = cfg.wv_idcs[len(cfg.wv_idcs)//2] if len(cfg.wv_idcs) > 1 else cfg.wv_idcs[0]
                        th = torch.sort(torch.unique(masks[wave_idx]), descending=True)[0][point_num-1] if point_num < len(torch.unique(masks)) else torch.inf
                        masks[masks<th] = 0
                    else:
                        # thresholding
                        masks[masks < cfg.nms_threshold] = 0
                        masks[masks > 0] -= cfg.nms_threshold
                else:
                    # cpu-based local maxima (time-consuming for large frames)
                    masks = regional_mask(outputs.squeeze().cpu().numpy(), th=cfg.nms_threshold)
                    masks = torch.tensor(masks, device=cfg.device)[None, ...]
                nms_time = time.process_time() - nms_start

                pts_start = time.process_time()
                wv_es_points = []
                for wv_idx in cfg.wv_idcs:
                    mask, output = (masks[wv_idx], outputs[wv_idx]) if len(cfg.wv_idcs) > 1 else (masks, outputs)
                    es_points, gt_points = align_points(mask, gt_pts, t_mat=t_mats[wv_idx], cfg=cfg, sr_img=output, stretch_opt=cfg.invivo)                    
                    wv_es_points.append(es_points)

                pts_time = time.process_time() - pts_start
                frame_time = time.process_time() - tic

                if len(cfg.wv_idcs) > 1:
                    # unravel list while adding wave and frame indices
                    wv_list = [np.vstack([el[0], k*np.ones(el[0].shape[-1]), len(dataset)*sequence + i*np.ones(el[0].shape[-1])]) 
                                            for k, el in enumerate(wv_es_points) if el[0].size > 0]
                    if len(wv_list) > 0:
                        pts = np.hstack(wv_list) if len(wv_list) > 1 else wv_list[0]
                        # fuse points using DBSCAN when eps > 0
                        es_points = [cluster_points(pts[:2].T, cluster_obj=cluster_obj).T] if pts.size > 0 and cfg.eps > 0 else [pts]
                else:
                    es_points = wv_es_points[0]
                    frame_idcs = len(dataset)*sequence + i*np.ones(wv_es_points[0][0].shape[-1])
                    pts = np.vstack([wv_es_points[0][0], np.zeros(wv_es_points[0][0].shape[-1]), frame_idcs])

                all_pts.append(es_points[0].T)
                all_pts_gt.append(gt_points[0].T)
                all_pts_indices.append(pts.T)
                
                if False:
                    import matplotlib.pyplot as plt
                    plt.figure()
                    plt.plot(*gt_points[0]*10, 'rx')
                    plt.plot(*es_points[0]*10, 'b+')
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
                        'frame': int(i) + sequence * dataset.frames_per_seq,
                    })

                # mean from bmode (skip for U-Net to reduce memory footprint)
                if not cfg.skip_bmode and not cfg.model.lower().__contains__('unet'):
                    bmode = batch[3] if cfg.input_type == 'rf' else batch[0]
                    bmode_frames.append(bmode)

                # create and upload ULM frame per sequence
                if (i+1) % dataset.frames_per_seq == 0:
                    if cfg.logging:
                        sres_ulm_img, velo_ulm_img = render_ulm_frame(all_pts, imgs, img_size, cfg, dataset.frames_per_seq, scale=cfg.upscale_factor)
                        sres_ulm_map = ulm_align(sres_ulm_img, gamma=cfg.gamma, cmap=cmap)
                        if velo_ulm_img.sum() > 0:
                            velo_ulm_map = np.zeros_like(velo_ulm_img)
                            velo_ulm_map[velo_ulm_img>0] = ulm_scale(velo_ulm_img[velo_ulm_img>0], gamma=cfg.gamma)
                            velo_ulm_map[velo_ulm_img<0] = ulm_scale(abs(velo_ulm_img[velo_ulm_img<0]), gamma=cfg.gamma)*-1
                            velo_ulm_map = img_color_map((velo_ulm_map+1)/2, cmap=velo_cmap)
                            wandb.log({"velo_ulm_img": wandb.Image(velo_ulm_map)})
                        bidx = imgs.shape[0] // 2
                        wandb.log({"magnitude_img": wandb.Image(imgs[bidx][0])})
                        wandb.log({"localization_img": wandb.Image(outputs[bidx][0])})
                        wandb.log({"sres_ulm_img": wandb.Image(sres_ulm_map)})
                        if cfg.synth_gt:
                            valid_pts = [p for p in all_pts_gt if p.size > 0]
                            sres_ulm_img = tracks2img(valid_pts, img_size=img_size, scale=cfg.upscale_factor, mode=cfg.track, fps=dataset.frames_per_seq)[0]
                            sres_ulm_map = ulm_align(sres_ulm_img, gamma=cfg.gamma, cmap=cmap)
                            wandb.log({"synth_ulm_img": wandb.Image(sres_ulm_map)})
                        if not cfg.skip_bmode and cfg.input_type == 'rf' and not cfg.model.lower().__contains__('unet'):
                            # averaging B-mode frames (skip for U-Net to reduce memory footprint)
                            sres_avg_img = np.nanmean(np.vstack(bmode_frames), axis=0)
                            sres_avg_img = sres_avg_img.sum(0) if len(sres_avg_img.shape) == 3 else sres_avg_img 
                            sres_avg_map = ulm_align(sres_avg_img, gamma=cfg.gamma, cmap=cmap)
                            wandb.log({"sres_avg_img": wandb.Image(sres_avg_map)})
                        if False:
                            # save b-mode frames as gif (for analysis purposes)
                            from utils.video_write import imageio_write_gif
                            frames = np.vstack(bmode_frames)[:, 0]
                            ret = imageio_write_gif(frames)

                pbar.update(i)

    errs = torch.tensor(ac_rmse_err)
    sres_rmse_mean = torch.nanmean(errs[..., 0], axis=0)
    sres_rmse_std = torch.std(errs[..., 0][~torch.isnan(errs[..., 0])], axis=0)
    print('Acc. Errors: %s' % str(torch.nanmean(errs, axis=0)))

    # remove empty arrays
    all_pts = [p for p in all_pts if len(p) > 0]

    # create and upload localizations as an artifact to wandb
    import h5py
    with h5py.File(f'localizations_{wandb.run.id}.h5', 'w') as hf:
        arr = np.vstack(all_pts_indices)
        h5obj = hf.create_dataset('localizations', data=arr, shape=arr.shape, compression='gzip', compression_opts=9, shuffle=True)
        h5obj.attrs['columns'] = ['x', 'z', 'amplitude', 'wave_index', 'frame_index']
        for k in cfg.keys():
            try:
                dataset.attrs[k] = data['config'][k]
            except:
                pass
    if cfg.logging:
        artifact = wandb.Artifact(f'localizations_{wandb.run.id}.h5', type='dataset')
        artifact.add_file(f'localizations_{wandb.run.id}.h5')
        wandb.log_artifact(artifact)

    # ground truth image
    all_pts_gt = [p for p in all_pts_gt if len(p) > 0] if len(all_pts_gt) > 0 else []
    if cfg.data_dir.lower().__contains__('insilico'):
        gtru_ulm_img, _ = tracks2img(all_pts_gt, img_size=img_size, scale=10, mode='all_in')
    else:
        gtru_ulm_img, _ = tracks2img(all_pts_gt, img_size=img_size, scale=cfg.upscale_factor, mode=cfg.track)
    
    # mean image
    sres_avg_img = np.nanmean(np.vstack(bmode_frames), axis=0) if not cfg.skip_bmode and cfg.input_type == 'rf' else np.zeros_like(gtru_ulm_img)
    sres_avg_img = sres_avg_img.sum(0) if len(sres_avg_img.shape) == 3 else sres_avg_img 

    # ULM frame
    sres_ulm_img, velo_ulm_img = render_ulm_frame(all_pts, imgs, img_size, cfg, dataset.frames_per_seq, scale=10 if cfg.data_dir.lower().__contains__('insilico') else cfg.upscale_factor)
    ssim_score = ssim(gtru_ulm_img[:, 2*cfg.upscale_factor:-2*cfg.upscale_factor], sres_ulm_img[:, 2*cfg.upscale_factor:-2*cfg.upscale_factor], data_range=sres_ulm_img.max()-sres_ulm_img.min())

    # gamma, sRGB gamma correction and color mapping
    sres_ulm_map = ulm_align(sres_ulm_img, gamma=cfg.gamma, cmap=cmap)
    gtru_ulm_map = ulm_align(gtru_ulm_img, gamma=cfg.gamma, cmap=cmap)
    sres_avg_map = ulm_align(sres_avg_img, gamma=cfg.gamma, cmap=cmap)
    hacc_ulm_map = ulm_align(normalize(h_acc), gamma=cfg.gamma, cmap=cmap) if h_acc is not None else np.zeros_like(sres_ulm_img)
    if cfg.logging:
        wandb.log({"sres_ulm_img": wandb.Image(sres_ulm_map)})
        wandb.log({"gtru_ulm_img": wandb.Image(gtru_ulm_map)})
        wandb.log({"sres_avg_img": wandb.Image(sres_avg_map)})
        wandb.log({"hacc_ulm_img": wandb.Image(hacc_ulm_map)})
        wandb.summary['Model'] = cfg.model
        wandb.summary['Type'] = cfg.input_type
        wandb.summary['TotalRMSE'] = sres_rmse_mean
        wandb.summary['TotalRMSEstd'] = sres_rmse_std
        wandb.summary['TotalJaccard'] = torch.nanmean(errs[..., 3], axis=0)
        wandb.summary['SSIM'] = ssim_score
        wandb.summary['TotalParameters'] = int(str(summary(model)).split('\n')[-3].split(' ')[-1].replace(',',''))
        wandb.save(str(Path('.') / 'logged_errors.csv'))
        wandb.finish()
