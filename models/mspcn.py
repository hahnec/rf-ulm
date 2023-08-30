# https://ieee-dataport.org/documents/us-data-vivo-rat-kidney

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class MSPCN(nn.Module):

    def __init__(self, upscale_factor, in_channels=1, semi_global_scale=1):
        super(MSPCN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, (9, 9), (1, 1), (4, 4))
        self.semi_global_block = SemiGlobalBlock2D(64, 64, semi_global_scale) if semi_global_scale != 1 else None

        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv6 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv7 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv8 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv9 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv10 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv11 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv12 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        
        self.conv13 = nn.Conv2d(64, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.semi_global_block(x) if self.semi_global_block is not None else x
        res1=x
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x=torch.add(res1,x)
        res3=x
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        x=torch.add(res3,x)
        res5=x
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        x=torch.add(res5,x)
        res7=x
        x = F.relu(self.conv8(x))
        x = self.conv9(x)
        x=torch.add(res7,x)
        res9=x
        x = F.relu(self.conv10(x))
        x = self.conv11(x)
        x=torch.add(res9,x)
        x = self.conv12(x)
        x=torch.add(res1,x)
        
        x = self.pixel_shuffle(self.conv13(x))
        
        return x

    def _initialize_weights(self):
        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv3.weight)
        init.orthogonal(self.conv4.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv5.weight)
        init.orthogonal(self.conv6.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv7.weight)
        init.orthogonal(self.conv8.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv9.weight)
        init.orthogonal(self.conv10.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv11.weight)
        init.orthogonal(self.conv12.weight)
        init.orthogonal(self.conv13.weight)


class SemiGlobalBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, sample_scale=2, kernel_size=5):
        super(SemiGlobalBlock2D, self).__init__()

        self.sample_scale = sample_scale
        self.feat_scale = max(1, sample_scale // 10)

        # Contracting path
        self.contract_conv = nn.Conv2d(in_channels, self.feat_scale * out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.contract_relu = nn.LeakyReLU()
        self.contract_pool = nn.MaxPool2d(kernel_size=sample_scale, stride=sample_scale)

        # Expanding path
        self.expand_conv = nn.Conv2d(self.feat_scale * out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.expand_relu = nn.LeakyReLU()
        self.expand_upsample = nn.Upsample(scale_factor=sample_scale, mode='nearest')

    def forward(self, x):
        # Contracting path
        x_scale = self.contract_conv(x)
        x_scale = self.contract_relu(x_scale)
        x_scale = self.contract_pool(x_scale)

        # Expanding path
        x_scale = self.expand_conv(x_scale)
        x_scale = self.expand_relu(x_scale)
        x_scale = self.expand_upsample(x_scale)

        # Adjust padding for correct output size
        padding_h = max(0, x.size(-2) - x_scale.size(-2))
        padding_w = max(0, x.size(-1) - x_scale.size(-1))
        x_scale = F.pad(x_scale, (padding_w // 2, padding_w // 2, padding_h // 2, padding_h // 2))

        # Skip connection via addition
        x = torch.add(x, x_scale)

        return x
