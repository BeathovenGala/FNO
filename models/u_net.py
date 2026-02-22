import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    """
    A lightweight U-Net for the MVC.
    Input: [B, 1, 64, 64]
    Output: [B, 1, 64, 64] (Kappa Map)
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=16):
        super().__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, base_filters)
        self.enc2 = self.conv_block(base_filters, base_filters*2)
        self.enc3 = self.conv_block(base_filters*2, base_filters*4)
        
        # MaxPool
        self.pool = nn.MaxPool2d(2)
        
        # Decoder
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self.conv_block(base_filters*4 + base_filters*2, base_filters*2)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(base_filters*2 + base_filters, base_filters)
        
        # Final
        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [B, 1, 64, 64]
        
        # Encode
        e1 = self.enc1(x)       # [B, 16, 64, 64]
        
        p1 = self.pool(e1)      # [B, 16, 32, 32]
        e2 = self.enc2(p1)      # [B, 32, 32, 32]
        
        p2 = self.pool(e2)      # [B, 32, 16, 16]
        e3 = self.enc3(p2)      # [B, 64, 16, 16] (Bottom)
        
        # Decode
        u2 = self.up2(e3)       # [B, 64, 32, 32]
        cat2 = torch.cat([u2, e2], dim=1) # [B, 96, 32, 32]
        d2 = self.dec2(cat2)    # [B, 32, 32, 32]
        
        u1 = self.up1(d2)       # [B, 32, 64, 64]
        cat1 = torch.cat([u1, e1], dim=1) # [B, 48, 64, 64]
        d1 = self.dec1(cat1)    # [B, 16, 64, 64]
        
        out = self.final(d1)    # [B, 1, 64, 64]
        
        # Force positive kappa (optional, but physically motivated)
        # Using Softplus to ensure positivity and differentiability
        return F.softplus(out) 
