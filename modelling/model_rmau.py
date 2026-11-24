import torch
import torch.nn as nn


class ResSeBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels_mid, out_channels_final, reduction: int = 6, ):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels_mid, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels_mid, affine=True), 
            nn.LeakyReLU(0.01, inplace=True),                  
            nn.Conv3d(out_channels_mid, out_channels_final, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels_final, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    
        hidden_channels = max(1, out_channels_final // reduction)
        self.global_pooling = nn.AdaptiveAvgPool3d(1) # N x C x D X H x W -> N x C x 1 x 1 x1
        self.fc3d = nn.Sequential(
            nn.Conv3d(out_channels_final, hidden_channels, kernel_size=1, bias=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(hidden_channels, out_channels_final, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        if in_channels != out_channels_final:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels_final, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels_final, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
            )
        else:
            self.shortcut = nn.Identity()

        self.final_act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.double_conv(x)

        se_weight = self.fc3d(self.global_pooling(out))
        out = out * se_weight

        res = self.shortcut(x)
        return self.final_act(out + res)


class Channel_Attention(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.max_pool_3d = nn.AdaptiveMaxPool3d(1)
        self.avg_pool_3d = nn.AdaptiveAvgPool3d(1)

        # Tanpa pembagian r: tetap di dimensi 'channels'
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=False),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(channels, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.shape

        mpool = self.max_pool_3d(x).view(b, c)
        avg_pool = self.avg_pool_3d(x).view(b, c)

        attn = self.sigmoid(self.shared_mlp(mpool) + self.shared_mlp(avg_pool))
        attn = attn.view(b, c, 1, 1, 1)
        return x * attn
    
class Spatial_Attention(nn.Module):
    def __init__(self, kernel_size: int = 7):
            super().__init__()
            assert kernel_size % 2 == 1, "kernel_size sebaiknya ganjil agar padding simetris"
            padding = kernel_size // 2

            # Input hasil concat [max_pool, avg_pool] punya 2 channel -> output 1 channel spatial map
            self.conv_3d = nn.Conv3d(
                in_channels=2,
                out_channels=1,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
            self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, C, D, H, W)
            # Pool di sepanjang dimensi channel (dim=1) -> masing-masing (B, 1, D, H, W)
            max_pool, _ = torch.max(x, dim=1, keepdim=True)
            avg_pool = torch.mean(x, dim=1, keepdim=True)

            # Gabungkan menjadi (B, 2, D, H, W) sesuai [Maxpool, AvgPool]
            feat = torch.cat([max_pool, avg_pool], dim=1)

            # Spatial attention map M_s: (B, 1, D, H, W)
            attn = self.sigmoid(self.conv_3d(feat))

            # Kalikan ke fitur input (B, C, D, H, W) via broadcasting
            return x * attn
        
class MAB3D(nn.Module):
    def __init__(self, in_channels, out_channels, rates=(1, 2, 4, 8), spatial_kernel_size=7):
        super().__init__()
        assert out_channels % len(rates) == 0
        bc = out_channels // len(rates)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, bc, kernel_size=3, padding=r, dilation=r, bias=False),
                nn.InstanceNorm3d(bc, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
            ) for r in rates
        ])
        self.channel_att = Channel_Attention(out_channels)
        self.spatial_att = Spatial_Attention(spatial_kernel_size)

    def forward(self, x):
        out = torch.cat([b(x) for b in self.branches], dim=1)
        return self.spatial_att(self.channel_att(out))
    
class ResSEMABUNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, features=(32, 64, 128, 256),
                 bottleneck=512, reduction=6):
        super().__init__()
        self.n_pool = len(features)
        self.enc = nn.ModuleList()
        prev = in_channels
        for f in features:
            self.enc.append(ResSeBlock3D(prev, f, f, reduction))
            prev = f
        self.pool = nn.MaxPool3d(2, 2)
        self.bottleneck = MAB3D(features[-1], bottleneck)

        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        prev = bottleneck
        for f in reversed(features):                              # 256, 128, 64, 32
            self.up.append(nn.ConvTranspose3d(prev, f, kernel_size=2, stride=2))
            self.dec.append(ResSeBlock3D(2 * f, f, f, reduction))  # 2f karena concat dengan skip
            prev = f

        self.final_mab = MAB3D(features[0], features[0])
        self.head = nn.Conv3d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        assert all(s % (2 ** self.n_pool) == 0 for s in x.shape[2:]), \
            f"ukuran {tuple(x.shape[2:])} harus kelipatan {2 ** self.n_pool}"
        skips = []
        for blk in self.enc:
            x = blk(x)
            skips.append(x)                                      
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([skip, up(x)], dim=1))
        return self.head(self.final_mab(x))                       # logit, belum softmax
