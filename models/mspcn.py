# https://ieee-dataport.org/documents/us-data-vivo-rat-kidney

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class MSPCN(nn.Module):

    def __init__(self, upscale_factor, in_channels=1):
        super(MSPCN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, (9, 9), (1, 1), (4, 4))
        
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
