import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels_mid, out_channels_final):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels_mid, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels_mid, affine=True),   # (1) BatchNorm -> InstanceNorm
            nn.LeakyReLU(0.01, inplace=True),                   # (2) ReLU -> LeakyReLU
            nn.Conv3d(out_channels_mid, out_channels_final, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels_final, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNET_Encoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = DoubleConv(1, 32, 64)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.block2 = DoubleConv(64, 64, 128)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.block3 = DoubleConv(128, 128, 256)
        self.pool3 = nn.MaxPool3d(2, 2)
        self.block4 = DoubleConv(256, 256, 256)

    def forward(self, x):
        skip1 = self.block1(x)
        skip2 = self.block2(self.pool1(skip1))
        skip3 = self.block3(self.pool2(skip2))
        bottleneck = self.block4(self.pool3(skip3))
        return bottleneck, skip1, skip2, skip3


class UNET_Decoder3D(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.upconv1 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)  # (3) 512 -> 256
        self.block1 = DoubleConv(512, 256, 256)                               #     concat: 256 + 256
        self.upconv2 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.block2 = DoubleConv(384, 128, 128)                               #     256 + 128
        self.upconv3 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.block3 = DoubleConv(192, 64, 64)                                 #     128 + 64
        self.final_conv = nn.Conv3d(64, num_classes, kernel_size=1)

    def forward(self, x, skip1, skip2, skip3):
        x = self.block1(torch.cat([skip3, self.upconv1(x)], dim=1))
        x = self.block2(torch.cat([skip2, self.upconv2(x)], dim=1))
        x = self.block3(torch.cat([skip1, self.upconv3(x)], dim=1))
        return self.final_conv(x)                                             # logit, belum softmax


class UNet3D(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.encoder = UNET_Encoder3D()
        self.decoder = UNET_Decoder3D(num_classes)

    def forward(self, x):
        bottleneck, s1, s2, s3 = self.encoder(x)
        return self.decoder(bottleneck, s1, s2, s3)


if __name__ == "__main__":
    m = UNet3D()
    x = torch.randn(1, 1, 24, 64, 64)
    print(tuple(m(x).shape), f"{sum(p.numel() for p in m.parameters()) / 1e6:.1f} juta parameter")