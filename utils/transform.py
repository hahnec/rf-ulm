## From https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py

import numpy as np
from PIL import Image
import random
import numbers

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from skimage import exposure


def pad_if_smaller(img, size, fill=0):
    ow, oh = img.size
    min_size = min(oh, ow)
    if min_size < size:
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomApply(object):
    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.p = p
        self.transforms = transforms

    def __call__(self, img, mask):
        if self.p < torch.rand(1):
            return img, mask
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target


class Resize(object):
    def __init__(self, size):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        target = F.resize(target, self.size, interpolation=T.InterpolationMode.NEAREST)
        return image, target

class RandomGrayscale(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, image, target):
        num_output_channels = F.get_image_num_channels(image)
        if random.random() < self.p:
            image = F.rgb_to_grayscale(image, num_output_channels=num_output_channels)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = np.ascontiguousarray(image[::-1])
            target = np.ascontiguousarray(target[::-1])
        
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = np.ascontiguousarray(image[::-1])
            target = np.ascontiguousarray(target[::-1])

        return image, target


class RandomCropTorch(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        h, w = image.shape[1], image.shape[2]
        pad_tb = max(0, self.size[0] - h)
        pad_lr = max(0, self.size[1] - w)
        image = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(image)
        mask = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 255)(mask)

        h, w = image.shape[1], image.shape[2]
        i = random.randint(0, h - self.size[0])
        j = random.randint(0, w - self.size[1])
        image = image[:, i:i + self.size[0], j:j + self.size[1]]
        mask = mask[:, i:i + self.size[0], j:j + self.size[1]]
        return image, mask

class RandomCrop(object):
    def __init__(self, size=None, upscale_factor=1):
        self.size = size if size is not None else (64, 64)
        self.upscale_factor = upscale_factor

    def __call__(self, img, gt):

        # convert back to PIL image
        pil_img = (np.dstack([img[0], img[1], img[0]]))/img.max() * 255
        pil_img = Image.fromarray(pil_img.astype('uint8'), 'RGB')
        pil_gt = gt[0]/gt.max()*255 if gt.max() != 0 else gt[0]
        pil_gt = Image.fromarray(pil_gt.astype('uint8'))

        # get crop coordinates
        i, j, h, w = T.RandomCrop.get_params(pil_img, output_size=self.size)

        # crop
        pil_img = T.functional.crop(pil_img, i, j, h, w)
        pil_gt = T.functional.crop(pil_gt, i*self.upscale_factor, j*self.upscale_factor, h*self.upscale_factor, w*self.upscale_factor)

        # convert back to numpy
        img = np.array(pil_img)[..., :2].swapaxes(2, 1).swapaxes(1, 0)
        gt = np.array(pil_gt)[None, ...] / np.array(pil_gt).max() if np.array(pil_gt).max() != 0 else np.array(pil_gt)[None, ...]

        return img, gt


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = F.to_tensor(target.astype(float)).long()
        # target = torch.as_tensor(np.array(target), dtype=torch.long)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RemoveWhitelines(object):
    def __call__(self, image, target):
        target = torch.where(target == 255, 0, target)
        return image, target


class RandomRotation(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return F.rotate(img, rotate_degree), F.rotate(mask, rotate_degree)


class GaussianBlur(object):
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img, mask):
        return F.gaussian_blur(img, self.kernel_size, self.sigma), mask


class ColorJitter(object):
    """Mostly from the docs"""
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def __call__(self, img, mask):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)

        return img, mask


class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, img, mask):
        return img + torch.randn(img.size()) * self.std + self.mean, mask

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CLAHE(object):
    def __call__(self, img: np.ndarray, target: np.ndarray):
        if type(img) == Image.Image:
            img = np.array(img)
        if np.argmin(img.shape) == 0:
            img = np.moveaxis(img, -1, 0)

        return exposure.equalize_adapthist(img), target


class NormalizeVol(object):
    def __init__(self):
        super(NormalizeVol, self).__init__()

    def __call__(self, waveform, *args, **kwargs):

        output = waveform/abs(waveform).max()

        if len(args) == 0 and len(kwargs) == 0:
            return output
        else:
            return output, *args, *kwargs


class ResizeBmode(object):
    def __init__(self):
        super(ResizeBmode, self).__init__()

    def __call__(self, bmode_frame, *args, **kwargs):
        
        output = bmode_frame

        if len(args) == 0 and len(kwargs) == 0:
            return output
        else:
            return output, *args, *kwargs

        return output, *args, *kwargs


if __name__ == '__main__':
    a = torch.rand((1, 3, 512, 512))
    # RandomRotate(20)(a, a)
    t = Compose([RandomApply([RandomRotate(30), GaussianNoise()]), ColorJitter(brightness=1)])
    t(a, a[:,0])

