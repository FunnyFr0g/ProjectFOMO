from clearml import Task, Logger
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
# from torch.utils.tensorboard import SummaryWriter
import time
from clearml import Dataset as CML_Dataset
from PIL import Image
# task = Task.init(
#     project_name='SmallObjectDetection',
#     task_name='FOMO-dronesOnly_train',
#     tags=['FOMO'])

USE_CLEARML = True

if USE_CLEARML:
    Task.ignore_requirements('pywin32')
    Task.add_requirements("networkx","3.4.2")
    task = Task.init(
            project_name='SmallObjectDetection',
            task_name='FOMO_56-doF_background_crop_train',
            tags=['FOMO'],
            reuse_last_task_id=True
            )
    task.execute_remotely(queue_name='default', exit_process=True)

# task.connect(params)



dataset_name = "drones_only_FOMO" #"FOMO-mva23" #

# coco_dataset = CML_Dataset.get(dataset_name="drones_only_FOMO", dataset_project="SmallObjectDetection")
# coco_dataset = CML_Dataset.get(dataset_id='45062c8b1fac490480d105ad9c945f22')
# dataset_path = coco_dataset.get_local_copy()

# dataset = CML_Dataset.get(dataset_name=dataset_name, dataset_project="SmallObjectDetection")
dataset = CML_Dataset.get(dataset_id='8ab8a51ea4304f20b272848a8b01a238')
dataset_path = dataset.get_local_copy()
# TRAIN_ANNOTATION_FILE = f"{dataset_path}/train/train_annotations/mva23_FOMO_train.json"
# TRAIN_IMAGE_DIR = f"{dataset_path}/train/images"
# VAL_ANNOTATION_FILE = f"{dataset_path}/val/val_annotations/mva23_FOMO_val.json"
# VAL_IMAGE_DIR = f"{dataset_path}/val/images"

TRAIN_ANNOTATION_FILE = f"{dataset_path}/train/train_annotations.json"
TRAIN_IMAGE_DIR = f"{dataset_path}/train/images"
VAL_ANNOTATION_FILE = f"{dataset_path}/val/val_annotations.json"
VAL_IMAGE_DIR = f"{dataset_path}/val/images"

# --- 1. Конфигурация ---

params = {
    "NUM_CLASSES" : 2,  # Кол-во классов (включая фон)
    "INPUT_SIZE" : (224, 224), # Размер входного изображения
    'BATCH_SIZE' : 64,
    "EPOCHS" : 150,
    "LR" : 1e-3,
    "trunkAt" : 4, # Номер слоя, где обрезать MobileNet. Для карты размером 56 это значение 4
    "NUM_WORKERS" : 1,
    "DATASET" : dataset.name,
    "DATASET_VERSION": dataset.version,
    "DATASET_ID" : dataset.id,
    }
if USE_CLEARML:
    params = task.connect(params)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

        try:
            image = cv2.imread(img_path) # крашит на кириллице
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            # print(f"Failed to convert {img_path} to RGB, ")
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)

        # Получаем аннотации для изображения
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Создаём маску классов (H, W)
        H, W, _ = image.shape
        mask = np.zeros((H, W), dtype=np.uint8)

        for ann in anns:
            if ann['category_id'] in self.cat_ids:
                class_id = self.cat_ids.index(ann['category_id']) + 1 # 0 - фон, 1 - птиц
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

        mask = augmented['mask']  # mask уже [H, W] = [224, 224]

        mask = torch.nn.functional.interpolate(
            mask.float().unsqueeze(0).unsqueeze(0),  # Добавляем batch и channel [1, 1, 224, 224]
            # size=(H//4, W//4),  # Новый размер
            size=(56, 56),  # Новый размер
            # size=(112, 112),  # Новый размер
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
    # A.Resize(params["INPUT_SIZE"][0], params["INPUT_SIZE"][1]),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# --- 3. Модель FOMO ---
class FomoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = mobilenet_v2(pretrained=True).features[:params['trunkAt']]  # Обрезаем MobileNetV2

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
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

    return running_loss / len(dataloader)

# --- 5. Основной цикл ---
def main():
    # Загрузка данных
    train_dataset = CocoDataset(TRAIN_ANNOTATION_FILE, TRAIN_IMAGE_DIR, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["BATCH_SIZE"],
        shuffle=True,
        num_workers=params['NUM_WORKERS'],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=1,
        multiprocessing_context='spawn'
    )

    val_dataset = CocoDataset(VAL_ANNOTATION_FILE, VAL_IMAGE_DIR, transform=transform)
    val_loader = DataLoader(val_dataset,
                            batch_size=params["BATCH_SIZE"],
                            shuffle=True, num_workers=params['NUM_WORKERS'],
                            pin_memory=True,
                            persistent_workers=False
                            )

    # Модель и оптимизатор
    model = FomoModel(params["NUM_CLASSES"]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["LR"])

    best_val_loss = float("inf")

    # Обучение
    for epoch in range(1, params["EPOCHS"]+1):
        epoch_start_time = time.time()
        train_loss = train(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        print(f"Epoch {epoch+1}/{params['EPOCHS']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Логирование метрик
        if USE_CLEARML:
            task.get_logger().report_scalar(
                title="Loss", series="Train", value=train_loss, iteration=epoch
            )
            task.get_logger().report_scalar(
                title="Loss", series="Val", value=val_loss, iteration=epoch
            )

        workdir = f'weights/FOMO_56_bg_crop_{dataset_name}_{params["DATASET_VERSION"]}'
        os.makedirs(workdir, exist_ok=True)

        # Сохранение весов
        if epoch != 0 and epoch%10 ==0:
            torch.save(model.state_dict(), os.path.join(workdir, f'{epoch}e.pth'))
            print("Model weights saved!")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(workdir, f'BEST_{epoch}e.pth'))





if __name__ == "__main__":
    main()