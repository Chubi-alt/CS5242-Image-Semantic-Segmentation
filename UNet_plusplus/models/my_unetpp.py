import torch
import torch.nn as nn

class VGGBlock(nn.Module):
    """ Standard Conv-BN-ReLU block used in UNet++ nodes. """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class MyUnetPlusPlus(nn.Module):
    """ 
    Hand-rolled UNet++ implementation.
    Designed to be compatible with existing training pipelines.
    """
    def __init__(self, in_channels=3, num_classes=32, features=[32, 128, 256, 512]):
        super().__init__()
        
        self.nb_filter = features # Number of filters per level

        # 1. Define Encoder Nodes (j=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.CONV0_0 = VGGBlock(in_channels, self.nb_filter[0])
        self.CONV1_0 = VGGBlock(self.nb_filter[0], self.nb_filter[1])
        self.CONV2_0 = VGGBlock(self.nb_filter[1], self.nb_filter[2])
        self.CONV3_0 = VGGBlock(self.nb_filter[2], self.nb_filter[3])

        # 2. Define Nested/Decoder Nodes (j>0)
        # Level 0 (Horizontal j=1, 2, 3)
        self.CONV0_1 = VGGBlock(self.nb_filter[0] + self.nb_filter[1], self.nb_filter[0])
        self.CONV0_2 = VGGBlock(self.nb_filter[0]*2 + self.nb_filter[1], self.nb_filter[0])
        self.CONV0_3 = VGGBlock(self.nb_filter[0]*3 + self.nb_filter[1], self.nb_filter[0])

        # Level 1 (Horizontal j=1, 2)
        self.CONV1_1 = VGGBlock(self.nb_filter[1] + self.nb_filter[2], self.nb_filter[1])
        self.CONV1_2 = VGGBlock(self.nb_filter[1]*2 + self.nb_filter[2], self.nb_filter[1])

        # Level 2 (Horizontal j=1)
        self.CONV2_1 = VGGBlock(self.nb_filter[2] + self.nb_filter[3], self.nb_filter[2])

        # Final Classification Head
        self.final = nn.Conv2d(self.nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder Path ---
        x0_0 = self.CONV0_0(x)
        x1_0 = self.CONV1_0(self.pool(x0_0))
        x2_0 = self.CONV2_0(self.pool(x1_0))
        x3_0 = self.CONV3_0(self.pool(x2_0))

        # --- Nested Skip Connections Path ---
        # Column j=1
        x0_1 = self.CONV0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.CONV1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.CONV2_1(torch.cat([x2_0, self.up(x3_0)], 1))

        # Column j=2
        x0_2 = self.CONV0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.CONV1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))

        # Column j=3
        x0_3 = self.CONV0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        # Final Output
        output = self.final(x0_3)
        return output