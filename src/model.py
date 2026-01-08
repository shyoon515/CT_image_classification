"""Modular residual CNN for CT Image Classification (from scratch)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Simple residual block with two 3x3 convs."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block (1x1 -> 3x3 -> 1x1)."""

    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        width = out_channels
        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


def make_layer(block: nn.Module, in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
    """Build one stage of residual blocks."""
    downsample = None
    if stride != 1 or in_channels != out_channels * block.expansion:
        downsample = nn.Sequential(
            conv1x1(in_channels, out_channels * block.expansion, stride),
            nn.BatchNorm2d(out_channels * block.expansion),
        )

    layers = [block(in_channels, out_channels, stride, downsample)]
    in_channels = out_channels * block.expansion
    for _ in range(1, blocks):
        layers.append(block(in_channels, out_channels))

    return nn.Sequential(*layers)


def build_mlp(in_features: int, hidden_sizes: list, num_classes: int, dropout: float) -> nn.Sequential:
    """Flexible MLP head."""
    layers = []
    prev = in_features
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, num_classes))
    return nn.Sequential(*layers)


class CTClassifier(nn.Module):
    """Configurable residual network for binary/multi-class CT classification.

    Args:
        block: Residual block class (BasicBlock or Bottleneck)
        layers: List of block counts per stage (e.g., [2,2,2,2])
        num_classes: Output classes. For binary classification, use 2.
        in_channels: Input channels (CT PNG는 3채널이므로 기본 3)
        stem_width: Stem conv output channels
        mlp_hidden: List of hidden sizes for MLP head
        dropout: Dropout prob in MLP head
    """

    def __init__(
        self,
        block: nn.Module = BasicBlock,
        layers: list = None,
        num_classes: int = 2,
        in_channels: int = 3,
        stem_width: int = 64,
        mlp_hidden: list = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if layers is None:
            layers = [2, 2, 2, 2]  # default ResNet-18 style
        if mlp_hidden is None:
            mlp_hidden = [256]

        self.inplanes = stem_width

        # Stem
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual stages (built dynamically based on layers list)
        self.stages = nn.ModuleList()
        in_channels = self.inplanes
        out_channels = stem_width
        for i, num_blocks in enumerate(layers):
            stride = 1 if i == 0 else 2
            stage = make_layer(block, in_channels, out_channels, num_blocks, stride)
            self.stages.append(stage)
            in_channels = out_channels * block.expansion
            out_channels *= 2  # double channels each stage after the first

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = in_channels
        self.head = build_mlp(feat_dim, mlp_hidden, num_classes, dropout)

        # Init weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for stage in self.stages:
            x = stage(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    def get_num_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }


def build_resnet18_scratch(num_classes: int = 2, **kwargs) -> CTClassifier:
    return CTClassifier(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes, **kwargs)


def build_resnet34_scratch(num_classes: int = 2, **kwargs) -> CTClassifier:
    return CTClassifier(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes, **kwargs)


def build_resnet50_scratch(num_classes: int = 2, **kwargs) -> CTClassifier:
    return CTClassifier(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes, **kwargs)
