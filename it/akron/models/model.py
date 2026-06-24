from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = self._down(base_channels, base_channels * 2)
        self.down2 = self._down(base_channels * 2, base_channels * 4, dropout=0.1)
        self.down3 = self._down(base_channels * 4, base_channels * 8, dropout=0.1)
        self.down4 = self._down(base_channels * 8, base_channels * 16, dropout=0.2)

        self.up1 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_channels * 16, base_channels * 8, dropout=0.1)
        self.up2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_channels * 8, base_channels * 4, dropout=0.1)
        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up4 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base_channels * 2, base_channels)
        self.output = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self._up(x5, x4, self.up1, self.conv1)
        x = self._up(x, x3, self.up2, self.conv2)
        x = self._up(x, x2, self.up3, self.conv3)
        x = self._up(x, x1, self.up4, self.conv4)
        return self.output(x)

    @staticmethod
    def _down(in_channels: int, out_channels: int, dropout: float = 0.0) -> nn.Sequential:
        return nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, dropout))

    @staticmethod
    def _up(x: torch.Tensor, skip: torch.Tensor, up: nn.Module, conv: nn.Module) -> torch.Tensor:
        x = up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return conv(torch.cat([skip, x], dim=1))


class ModelSummary:
    def build(self, model: nn.Module) -> str:
        lines = [
            "UNet model summary",
            "=" * 80,
            f"{'Layer':40} {'Type':24} {'Parameters':>12}",
            "-" * 80,
        ]
        total = 0
        trainable = 0
        for name, module in model.named_modules():
            if not name:
                continue
            parameters = sum(parameter.numel() for parameter in module.parameters(recurse=False))
            trainable_parameters = sum(
                parameter.numel() for parameter in module.parameters(recurse=False) if parameter.requires_grad
            )
            if parameters == 0:
                continue
            total += parameters
            trainable += trainable_parameters
            lines.append(f"{name:40} {module.__class__.__name__:24} {parameters:12,}")
        lines.extend(["-" * 80, f"Total parameters: {total:,}", f"Trainable parameters: {trainable:,}"])
        return "\n".join(lines)
