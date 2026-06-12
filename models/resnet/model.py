import torch.nn as nn

from models.registry import register_model


@register_model("resnet")
class SimpleResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
