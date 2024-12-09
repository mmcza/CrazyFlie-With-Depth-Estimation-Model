
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    #Squeeze-and-Excitation (SE) Block.

    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Squeeze: Global Average Pooling
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        weight = self.fc(x)
        return x * weight

class CBAM(nn.Module):

    #Convolutional Block Attention Module (CBAM).

    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        #moduł uwagi kanałowej
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Squeeze
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()  # Excitation
        )
        # Moduł uwagi przestrzennej
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca_weight = self.channel_attention(x)
        x = x * ca_weight
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Średnia na kanale
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Maksymalna wartość na kanale
        sa_weight = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * sa_weight
        return x
