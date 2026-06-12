import torch.nn as nn

from models.registry import register_model


@register_model("mobilenetv1")
class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
