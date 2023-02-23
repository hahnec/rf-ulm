import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pathlib import Path
from natsort import natsorted
import numpy as np
import scipy.io
from scipy.interpolate import interp1d, interp2d
from typing import Union, List, Tuple
from omegaconf import OmegaConf
import cv2

sample2dist = lambda x, c=345, fkHz=175.642, sample_rate=1: c/2 * x / sample_rate / fkHz
dist2sample = lambda d, c=345, fkHz=175.642, sample_rate=1: 2/c * d * fkHz * sample_rate

mat2dict = lambda mat: dict([(k[0], v.squeeze()) for v, k in zip(mat[0][0], list(mat.dtype.descr))])
bf_demod_100_bw2iq = lambda rf_100bw: rf_100bw[0::2, ...] - 1j*rf_100bw[1::2, ...]


class InSilicoDataset(Dataset):
    
    def __init__(
            self, 
            dataset_path = '', 
            transforms = None,
            rf_opt: bool = True,
            sequences: Union[List, Tuple] = None,
            rescale_factor: float = None,
            ch_gap: int = None,
            angle_threshold: float = None,
            blur_opt: bool = False,
            tile_opt: bool = False,
            ):

        torch.manual_seed(3008)

        self.dataset_path = Path(dataset_path) / 'RF' if rf_opt else Path(dataset_path) / 'IQ'
        print(self.dataset_path.resolve())

        self.transforms = transforms
        self.sequences = [0] if sequences is None else sequences
        self.rescale_factor = 1 if rescale_factor is None else rescale_factor
        self.ch_gap = 1 if ch_gap is None else ch_gap
        self.rf_opt = rf_opt if rf_opt is not None else rf_opt
        self.blur_opt = blur_opt if blur_opt is not None else blur_opt
        self.tile_opt = tile_opt if tile_opt is not None else tile_opt

        # exclude echoes from points at steep angles
        self.angle_threshold = angle_threshold if angle_threshold is not None else 1e9

        self.read_data()

    def read_data(self):

        self.seqns_filenames = natsorted([str(fname.name) for fname in self.dataset_path.iterdir() if str(fname.name).lower().endswith('.mat')])

        seq_mat = scipy.io.loadmat(str(self.dataset_path.parent / 'PALA_InSilicoFlow_sequence.mat'))

        frames_list = []
        labels_list = []
        mdatas_list = []
        for i in self.sequences:
            seq_frames, seq_labels, seq_mdatas = self.read_sequence(i)

            # sampling frequency (100% bandwidth mode of Verasonics) [Hz]
            seq_mdatas['fs'] = mat2dict(seq_mat['Receive'])['demodFrequency'] * 1e6

            # time between the emission and the beginning of reception [s]
            seq_mdatas['t0'] = 2*seq_mdatas['startDepth']*seq_mdatas['wavelength'] / seq_mdatas['c'] - mat2dict(seq_mat['TW'])['peak'] / seq_mdatas['f0']
            
            # collect sampling grid data
            p_data = mat2dict(seq_mat['PData'])
            seq_mdatas['Origin'] = p_data['Origin']
            seq_mdatas['Size'] = p_data['Size']
            seq_mdatas['PDelta'] = p_data['PDelta']

            seq_mdatas['param_x'] = (seq_mdatas['Origin'][0]+np.arange(seq_mdatas['Size'][1])*seq_mdatas['PDelta'][2])*seq_mdatas['wavelength']
            frames_list.append(seq_frames)
            labels_list.append(seq_labels)
            mdatas_list.append(seq_mdatas)
        
        # stack frames from different sequences
        self.all_frames = np.vstack(frames_list)
        self.all_labels = np.vstack(labels_list)
        self.all_metads = mdatas_list

    @staticmethod
    def get_vsource(mdata, tx_idx=0, beta = 1e-8):

        # extent of the phased-array
        width = mdata['xe'][-1] - mdata['xe'][0]    
        # virtual source (non-planar wave assumption)
        vsource = [
                    -width*np.cos(mdata['angles_list'][tx_idx]) * np.sin(mdata['angles_list'][tx_idx])/beta, 
                    -width*np.cos(mdata['angles_list'][tx_idx])**2/beta
                    ]

        return vsource, width

    def compose_config(self):

        # create outside metadata object
        args_dict = {"rescale_factor": self.rescale_factor, "ch_gap": self.ch_gap}
        meta_dict = {k: float(v) for k, v in self.all_metads[0].items() if not isinstance(v, np.ndarray)}
        cfg = OmegaConf.create({**args_dict, **meta_dict})

        return cfg

    def read_sequence(self, idx: int = 0):

        seq_mat = scipy.io.loadmat(self.dataset_path / self.seqns_filenames[idx])

        seq_frames = seq_mat['RFdata' if self.rf_opt else 'IQ'].swapaxes(0, -1).swapaxes(1, -1)
        seq_labels = seq_mat['ListPos'].swapaxes(0, -1)

        seq_mdatas = mat2dict(seq_mat['P'])

        # speed of sound [m/s]
        seq_mdatas['c'] = float(mat2dict(seq_mat['Resource'])['Parameters']['speedOfSound'])

        #central frequency [Hz]
        seq_mdatas['f0'] = mat2dict(seq_mat['Trans'])['frequency'] * 1e6

        # Wavelength [m]
        seq_mdatas['wavelength'] = seq_mdatas['c'] / seq_mdatas['f0']

        # x coordinates of transducer elements [m]
        seq_mdatas['xe'] = mat2dict(seq_mat['Trans'])['ElementPos'][:, 0]/1000

        tx_steer = mat2dict(seq_mat['TX'])['Steer']
        seq_mdatas['angles_list'] = np.array([tx_steer*1, tx_steer*0, tx_steer*-1, tx_steer*0])
        seq_mdatas['angles_list'] = seq_mdatas['angles_list'][:seq_mdatas['numTx'], 0]

        return seq_frames, seq_labels, seq_mdatas

    def compose_frame(self, rf_frame, metadata):

        rf_iq_frame = []
        # iterate over number of transmitted angles
        for i_tx in range(metadata['numTx']):

            # decompose stacked channels in sample domain for all angles
            idx_list = i_tx * metadata['NDsample'] + np.arange(metadata['NDsample'])
            rf_i = rf_frame[idx_list, :]

            # convert 100 bandwidth data to IQ signal ?
            rf_iq = bf_demod_100_bw2iq(rf_i)

            # resampling
            rf_iq = self.batched_iq2rf(rf_iq, mod_freq=metadata['fs'])

            rf_iq_frame.append(rf_iq)

        return np.array(rf_iq_frame)

    def batched_iq2rf(self, iq_data, mod_freq):

        x = np.linspace(0, len(iq_data)/mod_freq, num=len(iq_data), endpoint=True)
        t = np.linspace(0, len(iq_data)/mod_freq, num=int(len(iq_data)*self.rescale_factor), endpoint=True)
        
        f = interp1d(x, iq_data, axis=0)
        y = f(t)

        rf_data = y * np.exp(2*1j*np.pi*mod_freq*t[:, None])

        rf_data = 2**.5 * rf_data.real

        return rf_data

    @staticmethod
    def project_points_toa(points, metadata, vsource, width):

        # find transmit travel distances considering virtual source
        nonplanar_tdx = np.hypot((abs(vsource[0])-width/2)*(abs(vsource[0])>width/2), vsource[1])
        virtual_tdxs = np.hypot(points[0, ...]-vsource[0], points[2, ...]-vsource[1])
        dtxs = np.repeat((virtual_tdxs - nonplanar_tdx)[:, None], metadata['xe'].size, axis=1)

        # find receive travel distances
        drxs = np.hypot(points[0, ...][:, None]-np.repeat(metadata['xe'][None, :], points[0, ...].shape[0], axis=0), points[2, ...][:, None])

        # convert overall travel distances to travel times
        tau = (dtxs + drxs) / metadata['c']

        # convert travel times to sample indices (deducting blind zone?)
        sample_positions = (tau-metadata['t0']) * metadata['fs']

        return sample_positions

    def points2frame(self, points):
        
        gt_xz_coords = np.stack([points[0, ...], points[2, ...]]).T / self.all_metads[0]['wavelength'] - self.all_metads[0]['Origin'][::2]

        # upscale points and convert to integer
        sr_xz_coords = np.round(gt_xz_coords*self.rescale_factor).astype('int')

        # create image dimensions and mask outlying points
        ydim, xdim = self.all_frames[0].shape[-2:] * np.array([self.rescale_factor,  self.rescale_factor])
        mask = (sr_xz_coords[:, 1]>0) & (sr_xz_coords[:, 0]>0) & (sr_xz_coords[:, 1]<ydim) & (sr_xz_coords[:, 0]<xdim)

        # place upscaled points in image array
        frame_label = np.zeros([ydim, xdim])
        frame_label[sr_xz_coords[mask, 1], sr_xz_coords[mask, 0]] = 1

        return frame_label

    def crop_sloun(self, img, gt, pts):

        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(128, 128))
        img = transforms.functional.crop(img, i, j, h, w)
        gt = transforms.functional.crop(gt, i, j, h, w)
        shift_pts = pts - torch.tensor([i, j])
        pts = shift_pts[(shift_pts[:, 0] >= 0) & (shift_pts[:, 0] < h) & (shift_pts[:, 1] >= 0) & (shift_pts[:, 1] < w)]

        return img, gt, pts

    def __getitem__(self, idx):

        # load data frames
        frame_raw, label_raw, metadata = (self.all_frames[idx], self.all_labels[idx], self.all_metads[0])
        frame = self.compose_frame(frame_raw, metadata) if self.rf_opt else frame_raw

        # convert label data to ground-truth representation(s)
        nan_idcs = np.isnan(label_raw[0]) & np.isnan(label_raw[2])
        gt_points = label_raw[:, ~nan_idcs] * metadata['wavelength']
        
        if not self.rf_opt:

            # get rescaled ground-truth points
            gt_pts = self.rescale_factor * (label_raw[::2, :].T - metadata['Origin'][::2])[:, ::-1]
            
            # rescale frame
            hw = (self.rescale_factor * frame.shape[0], self.rescale_factor * frame.shape[1])
            frame = cv2.resize(abs(frame), hw[::-1], interpolation = cv2.INTER_CUBIC)
            
            # create ground-truth frame
            gt_frame = self.points2frame(gt_points)
            if self.blur_opt:
                gt_frame = cv2.GaussianBlur(gt_frame, (7, 7), 1)
                max_val = gt_frame.max() if gt_frame.max() != 0 else 1
                gt_frame = gt_frame / max_val
            
            # convert to torch tensor
            frame, gt_frame, gt_pts = torch.tensor(frame), torch.tensor(gt_frame), torch.tensor(gt_pts)
                
            # crop data to patch
            if self.tile_opt:
                frame, gt_frame, gt_pts = self.crop_sloun(frame, gt_frame, gt_pts)
            frame = frame.unsqueeze(0)
            gt_frame = gt_frame.unsqueeze(0)

            # adjust ground-truth points
            pad_pts = torch.nn.functional.pad(gt_pts, (0, 2-gt_pts.shape[1], 0, 50-gt_pts.shape[0]), "constant", float('NaN'))

            return frame, gt_frame, pad_pts#, gt_points

        gt_samples = []
        # iterate over plane waves
        for tx_idx in range(metadata['numTx']):

            # virtual source
            vsource, width = self.get_vsource(metadata, tx_idx=tx_idx)

            # project points to time-of-arrival
            sample_positions = self.project_points_toa(gt_points, metadata, vsource, width)

            # covnert sample to time unit
            time_position = sample_positions / metadata['fs']

            gt_samples.append(time_position)
        gt_samples = np.stack(gt_samples)

        if self.rf_opt:
            frame = frame.swapaxes(-2, -1)[:, ::self.ch_gap, ...]
            gt_samples = gt_samples.swapaxes(-2, -1)[:, ::self.ch_gap, ...]
            gt_mask = gt_mask[:, ::self.ch_gap, ...]
        else:
            frame = torch.nn.functional.interpolate(torch.tensor(abs(frame[None, None, :])), scale_factor=self.gt_upsample, mode='bicubic')[0]
            gt_frame = gt_frame[None, :]

        if self.transforms is not None:
            frame, gt_frame = self.transforms(frame, gt_frame)

        return frame, gt_frame, gt_samples, gt_points
    
    def __len__(self):
        return len(self.all_frames)
