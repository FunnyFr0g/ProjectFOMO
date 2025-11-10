import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import json
import os
from pycocotools.coco import COCO
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import random

my_seed = 42

torch.manual_seed(my_seed)
torch.cuda.manual_seed(my_seed)
np.random.seed(my_seed)
random.seed(my_seed)
torch.backends.cudnn.deterministic=True


# --- 1. Конфигурация ---
NUM_CLASSES = 2  # Кол-во классов (включая фон)
INPUT_SIZE = (224, 224)  # Размер входного изображения
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# trunkAt = 4 # Номер слоя, где обрезать MbileNet. Для карты размером 56 это значение 4

if not torch.cuda.is_available():
    print('#'*100, '\n CUDA IS NOT AVAILABLE')

# Пути к данным COCO
TRAIN_ANNOTATION_FILE = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train_FOMO\mva23_FOMO_train.json'
TRAIN_IMAGE_DIR = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train_FOMO\train\images'
VAL_ANNOTATION_FILE = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train_FOMO\mva23_FOMO_val.json.json'
VAL_IMAGE_DIR = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train_FOMO\val\images'

# --- 2. Датасет COCO ---
class CocoDataset(Dataset):
    def __init__(self, annotation_file, image_dir, transform=None):
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.image_dir = image_dir
        self.transform = transform
        self.cat_ids = self.coco.getCatIds()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Получаем аннотации для изображения
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Создаём маску классов (H, W)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)



        for ann in anns:
            if ann['category_id'] in self.cat_ids:
                class_id = self.cat_ids.index(ann['category_id']) + 1 # 0 - фон
                if 'segmentation' in ann:
                    # Если есть segmentation, используем его
                    mask += self.coco.annToMask(ann) * class_id
                else:
                    # Если нет - создаём маску из bbox
                    x, y, w, h = ann['bbox']
                    mask[int(y):int(y + h), int(x):int(x + w)] = class_id
                    # print(f'{class_id=}')

        # Применяем трансформации
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # В методе __getitem__ после transform:
        mask = augmented['mask']  # mask уже [H, W] = [224, 224]

        mask = torch.nn.functional.interpolate(
            mask.float().unsqueeze(0).unsqueeze(0),  # Добавляем batch и channel [1, 1, 224, 224]
            # size=(56, 56),  # Новый размер
            size=(112, 112),  # Новый размер
            mode='nearest'  # Без интерполяции (сохраняем целые классы)
        ).squeeze().long()  # Убираем batch и channel -> [56, 56]


        # import matplotlib.pyplot as plt
        # print(mask.sum())
        # print(img_path)
        # plt.imshow(mask.cpu().numpy())
        # plt.show()


        return image, mask.long()

# Трансформации
transform = A.Compose([
    A.Resize(INPUT_SIZE[0], INPUT_SIZE[1]),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# --- 3. Модель FOMO ---
class FomoBackbone56(nn.Module):
    def __init__(self):
        super().__init__()
        # self.mobilenet = mobilenet_v2(pretrained=True).features[:4]  # Обрезаем MobileNetV2
        mobilenet = mobilenet_v2(weights=None)
        state_dict = torch.load('models/mobilenet_v2_weights.pth', map_location='cpu')
        mobilenet.load_state_dict(state_dict)
        self.mobilenet = mobilenet.features[:4]  # Обрезаем MobileNetV2

    def forward(self, x):
        return self.mobilenet(x)


class FomoHead56(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Увеличиваем глубину feature map перед классификацией
        self.conv1 = nn.Conv2d(24, 48, kernel_size=3, padding=1)  # 24 -> 48 каналов
        self.act1 = nn.ReLU()

        # Дополнительный свёрточный слой (опционально)
        self.conv2 = nn.Conv2d(48, 32, kernel_size=3, padding=1)  # 48 -> 32
        self.act2 = nn.ReLU()

        # Финал: 1x1 свёртка для классификации
        self.conv3 = nn.Conv2d(32, num_classes, kernel_size=1)  # [num_classes, 56, 56]

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        return x

class FomoModel56(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = FomoBackbone56()
        self.head = FomoHead56(num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class ResidualBlock(nn.Module): # res-блоки для FOMO
    def __init__(self, channels, expansion=2, dropout=0.1):
        super(ResidualBlock, self).__init__()
        expanded_channels = channels * expansion

        self.conv1 = nn.Conv2d(channels, expanded_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_channels)
        self.act1 = nn.ReLU6(inplace=True)

        self.conv2 = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3,
                               padding=1, groups=expanded_channels, bias=False)  # depthwise
        self.bn2 = nn.BatchNorm2d(expanded_channels)
        self.act2 = nn.ReLU6(inplace=True)

        self.conv3 = nn.Conv2d(expanded_channels, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)

        out = out + residual  # skip connection
        out = nn.ReLU6(inplace=True)(out)

        return out


class FomoHeadResV0(nn.Module):
    def __init__(self, num_classes, num_blocks=2, dropout=0.1):
        super().__init__()

        # Первоначальное увеличение глубины
        self.initial_conv = nn.Conv2d(24, 48, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(48)
        self.initial_act = nn.ReLU6(inplace=True)

        # Стек residual-блоков
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(48, expansion=2, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # Промежуточное сжатие каналов
        self.mid_conv = nn.Conv2d(48, 32, kernel_size=1)
        self.mid_bn = nn.BatchNorm2d(32)
        self.mid_act = nn.ReLU6(inplace=True)

        # Финальная классификация
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

        # Инициализация весов
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_act(x)

        x = self.residual_blocks(x)

        x = self.mid_conv(x)
        x = self.mid_bn(x)
        x = self.mid_act(x)

        x = self.final_conv(x)
        return x

class FomoModelResV0(nn.Module):
    def __init__(self, num_classes=2, use_residual=True, num_res_blocks=2):
        super().__init__()
        self.backbone = FomoBackbone56()
        if use_residual:
            self.head = FomoHeadResV0(num_classes, num_blocks=num_res_blocks)
        else:
            self.head = FomoHead56(num_classes)  # оригинальная голова

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class FomoHeadResV1(nn.Module):
    def __init__(self, num_classes, num_blocks=3, dropout=0.1):
        super().__init__()

        # Стек residual-блоков
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(24, expansion=8+2*n, dropout=dropout) # 3 слоя последовательно расширяющихся
            for n in range(num_blocks)
        ])

        # Финальная классификация
        self.final_conv = nn.Conv2d(24, num_classes, kernel_size=1)

        # Инициализация весов
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.residual_blocks(x)

        x = self.final_conv(x)
        return x


class FomoModelResV1(nn.Module):
    def __init__(self, num_classes=2, use_residual=True, num_res_blocks=2):
        super().__init__()
        self.backbone = FomoBackbone56()
        if use_residual:
            self.head = FomoHeadResV1(num_classes, num_blocks=num_res_blocks)
        else:
            self.head = FomoHead56(num_classes)  # оригинальная голова

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)







# --- 3. Модель FOMO С картой 112x112 (срез на 2 слое) ---
class FomoBackbone112(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = mobilenet_v2(pretrained=True).features[:2]  # Обрезаем MobileNetV2

    def forward(self, x):
        return self.mobilenet(x)


class FomoHead112(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Увеличиваем глубину feature map перед классификацией
        self.conv1 = nn.Conv2d(16, 48, kernel_size=3, padding=1)  # 24 -> 48 каналов
        self.act1 = nn.ReLU()

        # Дополнительный свёрточный слой (опционально)
        self.conv2 = nn.Conv2d(48, 32, kernel_size=3, padding=1)  # 48 -> 32
        self.act2 = nn.ReLU()

        # Финал: 1x1 свёртка для классификации
        self.conv3 = nn.Conv2d(32, num_classes, kernel_size=1)  # [num_classes, 56, 56]

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        return x

class FomoModel112(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = FomoBackbone112()
        self.head = FomoHead112(num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)











# --- 4. Обучение ---
def train(model, dataloader, criterion, optimizer, device):
    log_dir = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\FOMO'
    writer = SummaryWriter(log_dir)
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)


        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(dataloader)

# --- 5. Основной цикл ---
def main():
    # Загрузка данных
    train_dataset = CocoDataset(TRAIN_ANNOTATION_FILE, TRAIN_IMAGE_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Модель и оптимизатор
    model = FomoModel112(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Обучение
    for epoch in range(1, EPOCHS+1):
        epoch_start_time = time.time()
        train_loss = train(model, train_loader, criterion, optimizer, DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}")
        if epoch != 0 and epoch%10 ==0:
            # Сохранение весов
            torch.save(model.state_dict(), f"FOMO_112_focalloss_{epoch}e_model_weights.pth")
            print("Model weights saved!")

if __name__ == "__main__":
    main()