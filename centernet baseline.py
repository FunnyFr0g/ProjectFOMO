import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair


class DeformConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, weight, bias, stride, padding, dilation, groups, deformable_groups, im2col_step):
        # Реализация forward pass DCN
        # Это упрощенная версия, нужно взять полную из mmcv
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass


class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(DeformConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, offset):
        # Использовать DeformConv2dFunction
        pass


class CTResNetNeck(nn.Module):
    def __init__(self, in_channel=512, num_deconv_filters=(256, 128, 64),
                 num_deconv_kernels=(4, 4, 4), use_dcn=True):
        super(CTResNetNeck, self).__init__()
        self.use_dcn = use_dcn

        # Часть 1: Деконволюционные слои (3 слоя)
        deconv_layers = []
        input_channels = in_channel
        for out_channels, kernel_size in zip(num_deconv_filters, num_deconv_kernels):
            # Пока ЗАГЛУШКА - обычная деконволюция. После установки mmcv замените на DCN.
            deconv = nn.ConvTranspose2d(
                input_channels, out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=1,
                bias=False)
            deconv_layers.append(deconv)
            deconv_layers.append(nn.BatchNorm2d(out_channels))
            deconv_layers.append(nn.ReLU(inplace=True))
            input_channels = out_channels

        # Часть 2: Дополнительные обычные сверточные слои (3 слоя)
        for _ in range(3):
            conv = nn.Conv2d(input_channels, input_channels,
                             kernel_size=3, stride=1, padding=1, bias=False)
            deconv_layers.append(conv)
            deconv_layers.append(nn.BatchNorm2d(input_channels))
            deconv_layers.append(nn.ReLU(inplace=True))

        # Объединяем все слои в одну последовательность
        self.deconv_layers = nn.Sequential(*deconv_layers)

        # ВАЖНО: Пока НЕ создавайте offset_layers. Мы добавим их только после установки mmcv.
        # self.offset_layers = nn.ModuleList() # Закомментируйте эту строку

    def forward(self, x):
        # Простой forward pass через Sequential
        return self.deconv_layers(x)


class CenterNetHead(nn.Module):
    def __init__(self, num_classes=1, in_channel=64, feat_channel=64):
        super(CenterNetHead, self).__init__()

        # Heatmap head (центры объектов)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Width-Height head (размеры bbox)
        self.wh_head = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, 2, kernel_size=1, stride=1, padding=0, bias=True)  # (w, h)
        )

        # Offset head (смещения)
        self.offset_head = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, 2, kernel_size=1, stride=1, padding=0, bias=True)  # (offset_x, offset_y)
        )

        # Инициализация
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        heatmap = self.heatmap_head(x)
        wh = self.wh_head(x)
        offset = self.offset_head(x)

        return heatmap, wh, offset


import torchvision.models as models


class CenterNet(nn.Module):
    def __init__(self, num_classes=1, use_dcn=True):
        super(CenterNet, self).__init__()
        # Загружаем полную resnet18
        resnet = models.resnet18(weights=None)  # Используйте weights=None, чтобы избежать warning

        # Разбираем backbone на именованные части, а не в Sequential
        self.backbone = nn.ModuleDict({
            'conv1': resnet.conv1,
            'bn1': resnet.bn1,
            'relu': resnet.relu,
            'maxpool': resnet.maxpool,
            'layer1': resnet.layer1,
            'layer2': resnet.layer2,
            'layer3': resnet.layer3,
            'layer4': resnet.layer4
        })

        # Остальная часть инициализации (neck, head) остается без изменений
        self.neck = CTResNetNeck(use_dcn=use_dcn)
        self.bbox_head = CenterNetHead(num_classes=num_classes)

    def forward(self, x):
        # Последовательно применяем слои backbone
        x = self.backbone['conv1'](x)
        x = self.backbone['bn1'](x)
        x = self.backbone['relu'](x)
        x = self.backbone['maxpool'](x)
        x = self.backbone['layer1'](x)
        x = self.backbone['layer2'](x)
        x = self.backbone['layer3'](x)
        x = self.backbone['layer4'](x)

        x = self.neck(x)
        heatmap, wh, offset = self.bbox_head(x)
        return heatmap, wh, offset


def load_mmdet_weights_to_torch(model, checkpoint_path):
    """Загружает веса из mmdet checkpoint в PyTorch модель"""

    # Загружаем чекпоинт mmdet
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Создаем mapping между именами слоев
    mapping = {}

    # Backbone mapping
    for key in state_dict.keys():
        if key.startswith('backbone.'):
            # Преобразуем имена mmdet в torch
            new_key = key.replace('backbone.', 'backbone.')

            # Особые случаи
            if 'downsample.0' in new_key:
                new_key = new_key.replace('downsample.0', 'downsample.0')
            elif 'downsample.1' in new_key:
                new_key = new_key.replace('downsample.1', 'downsample.1')

            mapping[key] = new_key

    # Neck mapping
    for key in state_dict.keys():
        if key.startswith('neck.'):
            # CTResNetNeck слои
            new_key = key.replace('neck.', 'neck.deconv_layers.')
            mapping[key] = new_key

    # Head mapping
    head_mapping = {
        'bbox_head.heatmap_head.0': 'bbox_head.heatmap_head.0',
        'bbox_head.heatmap_head.2': 'bbox_head.heatmap_head.2',
        'bbox_head.wh_head.0': 'bbox_head.wh_head.0',
        'bbox_head.wh_head.2': 'bbox_head.wh_head.2',
        'bbox_head.offset_head.0': 'bbox_head.offset_head.0',
        'bbox_head.offset_head.2': 'bbox_head.offset_head.2',
    }

    for old, new in head_mapping.items():
        for key in state_dict.keys():
            if key.startswith(old):
                suffix = key[len(old):]
                if suffix.startswith('.weight') or suffix.startswith('.bias'):
                    mapping[key] = new + suffix

    # Загружаем веса с преобразованием имен
    new_state_dict = {}
    for old_key, new_key in mapping.items():
        if old_key in state_dict:
            new_state_dict[new_key] = state_dict[old_key]

    # Загружаем в модель
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    return model


# Создаем модель
model = CenterNet(num_classes=1, use_dcn=True)

# Загружаем веса
model = load_mmdet_weights_to_torch(model, 'weights/centernet/latest.pth')

# Переводим в режим оценки
model.eval()

# Тестовый прогон
with torch.no_grad():
    dummy_input = torch.randn(1, 3, 512, 512)
    heatmap, wh, offset = model(dummy_input)
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"WH shape: {wh.shape}")
    print(f"Offset shape: {offset.shape}")