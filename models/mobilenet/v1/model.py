from typing import List

import torch
import torch.nn as nn

from models.registry import register_model


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


@register_model("mobilenetv1")
class MobileNetV1(nn.Module):

    def __init__(self, input_shape=(3, 32, 32), num_classes=10):
        super(MobileNetV1, self).__init__()

        filter_strides = [
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
            *[(512, 1) for _ in range(5)],
            (1024, 2),
            (1024, 2),
        ]

        in_channel = input_shape[0]
        self.init_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        in_channel = 32
        layers: List[DepthwiseSeparableBlock] = []
        for out_channel, stride in filter_strides:
            layers.append(
                DepthwiseSeparableBlock(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    stride=stride,
                )
            )
            in_channel = out_channel
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        with torch.no_grad():
            dummy = torch.zeros(64, *input_shape)
            dummy = self.init_conv(dummy)
            dummy = self.body(dummy)
            dummy = self.pool(dummy)

            in_channel = dummy.numel() // 64

        self.fc1 = nn.Linear(in_features=in_channel, out_features=num_classes)

    def forward(self, x):
        x = self.init_conv(x)

        x = self.body(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x
