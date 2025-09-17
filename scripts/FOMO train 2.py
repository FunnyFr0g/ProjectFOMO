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
from datetime import datetime

# --- 1. Конфигурация ---
NUM_CLASSES = 2  # Кол-во классов (включая фон)
INPUT_SIZE = (224, 224)  # Размер входного изображения
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trunkAt = 4  # Номер слоя, где обрезать MbileNet. Для карты размером 56 это значение 4

# Пути к данным COCO
TRAIN_ANNOTATION_FILE = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train_FOMO\mva23_FOMO_train.json'
TRAIN_IMAGE_DIR = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train_FOMO\train\images'
VAL_ANNOTATION_FILE = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train_FOMO\mva23_FOMO_val.json'
VAL_IMAGE_DIR = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train_FOMO\val\images'

# Директория для сохранения логов и чекпоинтов
LOG_DIR = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\FOMO'
CHECKPOINT_DIR = os.path.join(LOG_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


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
                class_id = self.cat_ids.index(ann['category_id']) + 1  # 0 - фон
                if 'segmentation' in ann:
                    # Если есть segmentation, используем его
                    mask += self.coco.annToMask(ann) * class_id
                else:
                    # Если нет - создаём маску из bbox
                    x, y, w, h = ann['bbox']
                    mask[int(y):int(y + h), int(x):int(x + w)] = class_id

        # Применяем трансформации
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = torch.nn.functional.interpolate(
            mask.float().unsqueeze(0).unsqueeze(0),  # Добавляем batch и channel [1, 1, 224, 224]
            size=(56, 56),  # Новый размер
            mode='nearest'  # Без интерполяции (сохраняем целые классы)
        ).squeeze().long()  # Убираем batch и channel -> [56, 56]

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


# --- 4. Функции обучения и валидации ---
def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)

    for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}")):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Логируем loss для каждого батча
        if writer is not None:
            writer.add_scalar('Train/Loss_per_batch', loss.item(), epoch * total_batches + batch_idx)

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def validate(model, dataloader, criterion, device, epoch, writer):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc=f"Validation Epoch {epoch}"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    val_loss /= len(dataloader)

    # Логируем validation loss
    if writer is not None:
        writer.add_scalar('Validation/Loss', val_loss, epoch)

    return val_loss


def save_checkpoint(model, optimizer, epoch, loss, val_loss, checkpoint_dir):
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"checkpoint_epoch_{epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    )

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': loss,
        'val_loss': val_loss,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


# --- 5. Основной цикл ---
def main():
    # Инициализация TensorBoard
    writer = SummaryWriter(LOG_DIR)

    # Загрузка данных
    train_dataset = CocoDataset(TRAIN_ANNOTATION_FILE, TRAIN_IMAGE_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = CocoDataset(VAL_ANNOTATION_FILE, VAL_IMAGE_DIR, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Модель и оптимизатор
    model = FomoModel(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Обучение
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()

        # Обучение на одной эпохе
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch, writer)

        # Логирование train loss
        writer.add_scalar('Train/Loss', train_loss, epoch)
        print(f"Epoch {epoch}/{EPOCHS}, Train Loss: {train_loss:.4f}")

        # Валидация каждые 10 эпох
        if epoch % 10 == 0 or epoch == EPOCHS:
            val_loss = validate(model, val_loader, criterion, DEVICE, epoch, writer)
            print(f"Epoch {epoch}/{EPOCHS}, Validation Loss: {val_loss:.4f}")

            # Сохранение чекпоинта
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, CHECKPOINT_DIR)

            # Сохранение лучшей модели
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(CHECKPOINT_DIR, f"best_model_epoch_{epoch}.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with val_loss: {val_loss:.4f}")

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds\n")

    # Закрытие writer
    writer.close()


if __name__ == "__main__":
    main()