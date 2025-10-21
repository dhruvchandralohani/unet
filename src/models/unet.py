import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DoubleConv(nn.Module):
    """A double convolution block: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mid_channels (Optional[int]): Number of channels after the first convolution.
            If None, defaults to out_channels. Defaults to None.
        dropout_prob (float): Dropout probability for regularization. Defaults to 0.0.
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 mid_channels: Optional[int] = None, dropout_prob: float = 0.0) -> None:
        super().__init__()
        mid_channels = mid_channels or out_channels
        
        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        
        if dropout_prob > 0:
            layers.append(nn.Dropout2d(p=dropout_prob))
            
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the double convolution block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block for U-Net encoder: MaxPool -> DoubleConv.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dropout_prob (float): Dropout probability for DoubleConv. Defaults to 0.0.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_prob: float = 0.0) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels, dropout_prob=dropout_prob)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the downsampling block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height//2, width//2).
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block for U-Net decoder: Upsample/ConvTranspose -> Concatenate -> DoubleConv.

    Args:
        in_channels (int): Number of input channels (including skip connection channels).
        out_channels (int): Number of output channels.
        bilinear (bool): If True, uses bilinear upsampling; else, uses transposed convolution.
            Defaults to True.
        dropout_prob (float): Dropout probability for DoubleConv. Defaults to 0.0.
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 bilinear: bool = True, dropout_prob: float = 0.0) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, mid_channels=in_channels // 2, dropout_prob=dropout_prob
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels, dropout_prob=dropout_prob)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass through the upsampling block.

        Args:
            x1 (torch.Tensor): Input tensor from previous layer, shape (batch_size, in_channels//2, height, width).
            x2 (torch.Tensor): Skip connection tensor, shape (batch_size, in_channels//2, height*2, width*2).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height*2, width*2).
        """
        x1 = self.up(x1)
        # Adjust spatial dimensions for concatenation
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_y or diff_x:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, 
                           diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture for semantic segmentation with encoder-decoder and skip connections.

    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB). Defaults to 3.
        out_channels (int): Number of output channels (e.g., 1 for binary segmentation). Defaults to 1.
        bilinear (bool): If True, uses bilinear upsampling in decoder. Defaults to True.
        dropout_prob (float): Dropout probability for all DoubleConv blocks. Defaults to 0.0.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, 
                 bilinear: bool = True, dropout_prob: float = 0.0) -> None:
        super().__init__()
        self.encoder1 = DoubleConv(in_channels, 64, dropout_prob=dropout_prob)
        self.encoder2 = Down(64, 128, dropout_prob=dropout_prob)
        self.encoder3 = Down(128, 256, dropout_prob=dropout_prob)
        self.encoder4 = Down(256, 512, dropout_prob=dropout_prob)
        self.bottleneck = Down(512, 1024, dropout_prob=dropout_prob)
        
        self.decoder1 = Up(1024 + 512, 512, bilinear=bilinear, dropout_prob=dropout_prob)
        self.decoder2 = Up(512 + 256, 256, bilinear=bilinear, dropout_prob=dropout_prob)
        self.decoder3 = Up(256 + 128, 128, bilinear=bilinear, dropout_prob=dropout_prob)
        self.decoder4 = Up(128 + 64, 64, bilinear=bilinear, dropout_prob=dropout_prob)
        
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the U-Net model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, out_channels, height, width).
        """
        # Encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.bottleneck(x4)
        
        # Decoder with skip connections
        x = self.decoder1(x5, x4)
        x = self.decoder2(x, x3)
        x = self.decoder3(x, x2)
        x = self.decoder4(x, x1)
        
        # Final convolution
        logits = self.out_conv(x)
        return logits