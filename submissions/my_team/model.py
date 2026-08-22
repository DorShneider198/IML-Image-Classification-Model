"""Small ResNet-style CNN trained from scratch for the 20-class bird subset.

Everything is defined here: no torchvision.models, no pretrained weights, no
downloads. The stem keeps full resolution instead of the usual 7x7 stride-2 +
maxpool, because the classes include several visually similar groups (parrots,
waterfowl, waders) where fine detail carries the signal.
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """conv3x3 -> BN -> ReLU -> conv3x3 -> BN, plus a skip, then ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Project the skip only when the shapes would not line up.
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return self.relu(out + identity)


class ModelArchitecture(nn.Module):
    """Student model.

    input:  torch.Tensor of shape [batch_size, 3, 224, 224]
    output: torch.Tensor of shape [batch_size, num_classes] logits
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()

        # Detail-preserving stem: 3x3 stride 1, no maxpool.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 32 -> 64 -> 128 -> 256, two blocks per stage, each stage halving.
        self.stage1 = self._make_stage(32, 32, stride=2)
        self.stage2 = self._make_stage(32, 64, stride=2)
        self.stage3 = self._make_stage(64, 128, stride=2)
        self.stage4 = self._make_stage(128, 256, stride=2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        stride: int,
        blocks: int = 2,
    ) -> nn.Sequential:
        """Two BasicBlocks; the first carries the stride and channel change."""
        layers = [BasicBlock(in_channels, out_channels, stride=stride)]
        layers.extend(
            BasicBlock(out_channels, out_channels, stride=1)
            for _ in range(blocks - 1)
        )
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return self.head(x)


if __name__ == "__main__":
    model = ModelArchitecture()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        feature = model.stem(x)
        print(f"\n{'stem':<8} {tuple(feature.shape)}")

        for stage in ["stage1", "stage2", "stage3", "stage4"]:
            feature = getattr(model, stage)(feature)
            print(f"{stage:<8} {tuple(feature.shape)}")

    logits = model(x)

    print(f"\nOutput shape: {tuple(logits.shape)}")
    assert logits.shape == (2, 20), f"expected (2, 20), got {tuple(logits.shape)}"
    print("OK")
