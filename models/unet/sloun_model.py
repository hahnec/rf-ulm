""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from .sloun_parts import *


class SlounUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(SlounUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (SlounDoubleConv(n_channels, 64))
        self.down1 = (SlounDown(64, 128))
        self.down2 = (SlounDown(128, 256))
        self.down3 = (SlounDown(256, 512))
        self.latent_conv = (SlounLatent(512, 512))
        factor = 2 if bilinear else 1
        self.up1 = (SlounUp(512, 256 // factor, bilinear))
        self.up2 = (SlounUp(256, 128 // factor, bilinear))
        self.up3 = (SlounUp(128, 64, bilinear))
        self.outc = (SlounOutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.latent_conv(x4)
        x = self.up1(x5, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.outc = torch.utils.checkpoint(self.outc)
