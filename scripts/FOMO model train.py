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

# --- 1. Конфигурация ---
NUM_CLASSES = 2  # Кол-во классов (включая фон)
INPUT_SIZE = (224, 224)  # Размер входного изображения
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trunkAt = 4 # Номер слоя, где обрезать MbileNet. Для карты размером 56 это значение 4

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
class FomoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = mobilenet_v2(pretrained=True).features[:trunkAt]  # Обрезаем MobileNetV2

    def forward(self, x):
        return self.mobilenet(x)


class FomoHead(nn.Module):
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

class FomoModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = FomoBackbone()
        self.head = FomoHead(num_classes)

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
    model = FomoModel(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Обучение
    for epoch in range(1, EPOCHS+1):
        epoch_start_time = time.time()
        train_loss = train(model, train_loader, criterion, optimizer, DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}")
        if epoch != 0 and epoch%10 ==0:
            # Сохранение весов
            torch.save(model.state_dict(), f"FOMO_56_focalloss_{epoch}e_model_weights.pth")
            print("Model weights saved!")

if __name__ == "__main__":
    main()