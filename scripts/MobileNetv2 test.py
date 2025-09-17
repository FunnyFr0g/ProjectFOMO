import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

model = mobilenet_v2(pretrained=True)
print(model.features)  # Вывод всех слоёв бэкбона

truncate_at = 2
imsize = 224

class TruncatedMobileNetV2(nn.Module):
    def __init__(self, truncate_at=6):
        super().__init__()
        backbone = mobilenet_v2(pretrained=True).features
        self.truncated = nn.Sequential(*list(backbone.children())[:truncate_at])

    def forward(self, x):
        return self.truncated(x)

model = TruncatedMobileNetV2(truncate_at)

dummy_input = torch.randn(1, 3, imsize, imsize)
print(model(dummy_input).shape)  # Например, [1, 64, 14, 14]