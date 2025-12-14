import os
import random
import numpy as np
from PIL import Image
# import cv2
from pathlib import Path
from collections import defaultdict
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import GradScaler, autocast
import warnings

from clearml import Task, Logger, Dataset as CML_Dataset


warnings.filterwarnings('ignore')

USE_CLEARML = True

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if USE_CLEARML:
    Task.ignore_requirements('pywin32')
    Task.add_requirements("networkx","3.4.2")
    task = Task.init(
            project_name='SmallObjectDetection',
            task_name='TempFilt 3d conv',
            tags=['TemporalFiltration'],
            reuse_last_task_id=True
            )
    task.execute_remotely(queue_name='default', exit_process=True)



# Конфигурация
class Config:
    # Пути
    if USE_CLEARML:
        ds = CML_Dataset.get(dataset_name='bird_drones_classification', dataset_project='SmallObjectDetection',)
        data_dir = ds.get_local_copy()
    else:
        data_dir = r"M:\object_crops"  # Измените на свой путь

    model_save_dir = r"weights/3d_cnn_classifier"
    log_dir = r"weights/3d_cnn_classifier/logs"

    # Параметры данных
    img_size = (32, 32)  # Размер кадра для 3D CNN (можно увеличить)
    sequence_length = 5  # Количество кадров в последовательности
    batch_size = 8  # Меньше из-за 3D CNN
    num_workers = 4

    # Параметры 3D CNN
    hidden_dim = 128
    dropout_rate = 0.4

    # Параметры обучения
    epochs = 100
    learning_rate = 0.001
    weight_decay = 1e-4
    gradient_clip = 1.0

    # Augmentation
    use_3d_augmentation = True

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def create_dirs(cls):
        os.makedirs(cls.model_save_dir, exist_ok=True)
        os.makedirs(cls.log_dir, exist_ok=True)


# Создание директорий
Config.create_dirs()


# ==================== DATASET (обновленная версия) ====================
class VideoSequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None, sequence_length=16, is_train=True,
                 split_ratio=0.8, seed=42, overlap=4):
        """
        Dataset для видеопоследовательностей с дронами и птицами.

        Args:
            root_dir: путь к данным
            transform: трансформации изображений
            sequence_length: количество кадров в последовательности
            is_train: флаг обучения/валидации
            split_ratio: доля данных для обучения
            seed: фиксированный seed для воспроизводимости
            overlap: перекрытие между последовательностями
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.sequence_length = sequence_length
        self.is_train = is_train
        self.overlap = overlap

        # Собираем данные
        self.sequences = []
        self.labels = []  # 0 - птица, 1 - дрон
        self.identities = []  # Уникальные ID для каждого объекта

        self._load_data(split_ratio, seed)

        logger.info(f"{'Train' if is_train else 'Val'}: {len(self.sequences)} sequences")
        if len(self.labels) > 0:
            logger.info(f"Classes: Birds: {(np.array(self.labels) == 0).sum()}, "
                        f"Drones: {(np.array(self.labels) == 1).sum()}")
        logger.info(f"Unique identities: {len(set(self.identities))}")

    def _load_data(self, split_ratio, seed):
        """Загрузка и разделение данных по идентификаторам объектов"""
        all_objects = []  # Список всех объектов (папок)
        object_labels = []  # Метки объектов
        object_images = []  # Списки изображений для каждого объекта

        # Проходим по всем папкам и собираем информацию об объектах
        for folder_idx, folder in enumerate(sorted(self.root_dir.iterdir())):
            if not folder.is_dir():
                continue

            # Определяем класс по имени папки
            folder_name = folder.name.upper()
            if "DRONE" in folder_name:
                label = 1  # Дрон
            elif "BIRD" in folder_name:
                label = 0  # Птица
            else:
                continue  # Пропускаем папки без меток

            # Собираем все изображения в папке
            images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
            if len(images) < self.sequence_length:
                logger.warning(f"Папка {folder.name} содержит недостаточно изображений: {len(images)}")
                continue

            all_objects.append(folder.name)
            object_labels.append(label)
            object_images.append(images)

        if not all_objects:
            logger.error("Не найдено объектов для обучения!")
            return

        # Разделяем объекты (не последовательности) на train/val
        train_objects, val_objects, train_labels, val_labels, train_images, val_images = train_test_split(
            all_objects, object_labels, object_images,
            test_size=1 - split_ratio,
            random_state=seed,
            stratify=object_labels
        )

        # Создаем последовательности для выбранного сета
        if self.is_train:
            objects = train_objects
            labels = train_labels
            images_list = train_images
        else:
            objects = val_objects
            labels = val_labels
            images_list = val_images

        # Создаем последовательности для каждого объекта
        for obj_idx, (obj_name, label, images) in enumerate(zip(objects, labels, images_list)):
            # Уникальный ID для этого объекта
            obj_id = obj_idx

            # Создаем последовательности с заданным перекрытием
            step = max(1, self.sequence_length - self.overlap)
            for start_idx in range(0, len(images) - self.sequence_length + 1, step):
                seq_images = images[start_idx:start_idx + self.sequence_length]

                # Добавляем только если получили полную последовательность
                if len(seq_images) == self.sequence_length:
                    self.sequences.append(seq_images)
                    self.labels.append(label)
                    self.identities.append(obj_id)

        logger.info(f"Created {len(self.sequences)} sequences from {len(objects)} objects")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """Возвращает одну видеопоследовательность"""
        seq_paths = self.sequences[idx]
        label = self.labels[idx]
        identity = self.identities[idx]

        # Загружаем и обрабатываем все кадры последовательности
        frames = []
        for img_path in seq_paths:
            try:
                # Загружаем изображение
                img = Image.open(img_path).convert('RGB')

                # Применяем трансформации
                if self.transform:
                    img = self.transform(img)

                frames.append(img)
            except Exception as e:
                logger.error(f"Ошибка загрузки {img_path}: {e}")
                # Заполняем нулями в случае ошибки
                frames.append(torch.zeros(3, Config.img_size[0], Config.img_size[1]))

        # Собираем все кадры в один тензор
        if len(frames) > 0:
            # Получаем размеры: (sequence_length, C, H, W)
            frames_tensor = torch.stack(frames, dim=0)
        else:
            frames_tensor = torch.zeros(self.sequence_length, 3,
                                        Config.img_size[0], Config.img_size[1])

        # Для 3D CNN нам нужно изменить размерность: (C, T, H, W)
        # где T = sequence_length
        frames_tensor = frames_tensor.permute(1, 0, 2, 3)  # (C, T, H, W)

        return {
            'video': frames_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'identity': torch.tensor(identity, dtype=torch.long),
            'sequence_id': idx
        }


# ==================== AUGMENTATIONS для 3D CNN ====================
class VideoTransform:
    """Трансформации для видеопоследовательностей"""

    @staticmethod
    def get_train_transforms():
        """Трансформации для обучения"""
        return transforms.Compose([
            transforms.Resize(Config.img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_val_transforms():
        """Трансформации для валидации"""
        return transforms.Compose([
            transforms.Resize(Config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


# ==================== 3D CNN MODELS ====================
class Simple3DCNN(nn.Module):
    """Простая 3D CNN архитектура"""

    def __init__(self, num_classes=2, dropout_rate=0.4):
        super().__init__()

        # Первый блок сверток
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )

        # Второй блок сверток
        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )

        # Третий блок сверток
        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )

        # Четвертый блок сверток
        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )

        # Глобальный average pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

        # Инициализация весов
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: тензор формы (batch_size, 3, T, H, W)
        где T = sequence_length
        """
        # 3D свертки
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Глобальный pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Классификация
        x = self.classifier(x)

        return x


class ResNet3D(nn.Module):
    """3D ResNet-like архитектура"""

    def __init__(self, num_classes=2, dropout_rate=0.4):
        super().__init__()

        # Первый слой
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                               padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # Residual блоки
        self.layer1 = self._make_layer(32, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        # Глобальный average pooling
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        # Классификатор
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []

        # Первый блок со страйдом
        layers.append(ResidualBlock3D(in_channels, out_channels, stride))

        # Остальные блоки
        for _ in range(1, blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResidualBlock3D(nn.Module):
    """3D Residual Block"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class R3D_18(nn.Module):
    """3D ResNet-18 архитектура"""

    def __init__(self, num_classes=2, dropout_rate=0.4):
        super().__init__()

        # Используем 2D ResNet как основу и адаптируем для 3D
        resnet2d = models.resnet18(pretrained=True)

        # Адаптация первого слоя для 3D
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7),
                               stride=(1, 2, 2), padding=(1, 3, 3), bias=False)

        # Копируем веса из 2D ResNet (усредняем по временной оси)
        with torch.no_grad():
            weight_2d = resnet2d.conv1.weight.data
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            weight_3d = weight_3d / 3  # Нормализация
            self.conv1.weight.data = weight_3d

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # Адаптация остальных слоев
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(dropout_rate)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []

        # Первый блок
        layers.append(BasicBlock3D(in_channels, out_channels, stride))

        # Остальные блоки
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class BasicBlock3D(nn.Module):
    """Basic 3D Block для ResNet"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


# ==================== TRAINER ====================
class VideoClassifierTrainer:
    def __init__(self, config, model_type='simple'):
        self.config = config
        self.device = config.device
        self.model_type = model_type

        # Инициализация модели
        self.model = self._create_model(model_type).to(self.device)

        # Функция потерь
        self.criterion = nn.CrossEntropyLoss()

        # Оптимизатор
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5,
        )

        # Mixed precision training
        self.scaler = GradScaler() if self.device.type == 'cuda' else None

        # Загрузка данных
        self._setup_data()

        # История обучения
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rate': []
        }

        # Лучшая модель
        self.best_val_accuracy = 0.0
        self.best_val_loss = float('inf')

        logger.info(f"Model initialized: {model_type}")
        logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _create_model(self, model_type):
        """Создание модели по типу"""
        if model_type == 'simple':
            return Simple3DCNN(num_classes=2, dropout_rate=self.config.dropout_rate)
        elif model_type == 'resnet3d':
            return ResNet3D(num_classes=2, dropout_rate=self.config.dropout_rate)
        elif model_type == 'r3d_18':
            return R3D_18(num_classes=2, dropout_rate=self.config.dropout_rate)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _setup_data(self):
        """Загрузка и подготовка данных"""
        # Фиксированный seed для воспроизводимости
        seed = 42

        # Трансформации
        train_transform = VideoTransform.get_train_transforms()
        val_transform = VideoTransform.get_val_transforms()

        # Датасеты
        self.train_dataset = VideoSequenceDataset(
            root_dir=self.config.data_dir,
            transform=train_transform,
            sequence_length=self.config.sequence_length,
            is_train=True,
            split_ratio=0.8,
            seed=seed,
            overlap=0
        )

        self.val_dataset = VideoSequenceDataset(
            root_dir=self.config.data_dir,
            transform=val_transform,
            sequence_length=self.config.sequence_length,
            is_train=False,
            split_ratio=0.8,
            seed=seed,
            overlap=0
        )

        # Проверка данных
        if len(self.train_dataset) == 0 or len(self.val_dataset) == 0:
            logger.error("Нет данных для обучения!")
            raise ValueError("Пустые датасеты")

        # DataLoader'ы
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=min(self.config.num_workers, 4),
            pin_memory=True,
            drop_last=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=min(self.config.num_workers, 4),
            pin_memory=True
        )

        logger.info(f"Train: {len(self.train_dataset)} sequences")
        logger.info(f"Val: {len(self.val_dataset)} sequences")

        # Информация о классах
        train_labels = np.array(self.train_dataset.labels)
        val_labels = np.array(self.val_dataset.labels)

        logger.info(f"Train - Birds: {(train_labels == 0).sum()}, Drones: {(train_labels == 1).sum()}")
        logger.info(f"Val - Birds: {(val_labels == 0).sum()}, Drones: {(val_labels == 1).sum()}")

    def train_epoch(self, epoch):
        """Одна эпоха обучения"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Перемещаем данные на устройство
            videos = batch['video'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)

            # Mixed precision training
            with autocast(enabled=self.device.type == 'cuda'):
                # Forward pass
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()

            # Статистика
            total_loss += loss.item()

            # Точность
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Логирование каждые 10 батчей
            if batch_idx % 10 == 0:
                accuracy = (predicted == labels).float().mean().item()
                logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | "
                            f"Loss: {loss.item():.4f} | Acc: {accuracy:.4f}")

        # Средние значения за эпоху
        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples

        self.history['train_loss'].append(avg_loss)
        self.history['train_acc'].append(accuracy)
        self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

        return avg_loss, accuracy

    def validate(self, epoch):
        """Валидация"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        all_labels = []
        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Перемещаем данные на устройство
                videos = batch['video'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)

                # Forward pass
                outputs = self.model(videos)
                loss = self.criterion(outputs, labels)

                # Статистика
                total_loss += loss.item()

                # Точность
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                # Для метрик
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

        # Средние значения
        avg_loss = total_loss / max(len(self.val_loader), 1)
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        self.history['val_loss'].append(avg_loss)
        self.history['val_acc'].append(accuracy)

        # Обновление scheduler
        self.scheduler.step(avg_loss)

        # Сохраняем лучшую модель
        if accuracy > self.best_val_accuracy:
            self.best_val_accuracy = accuracy
            self.best_val_loss = avg_loss
            self.save_model(f"best_model_epoch_{epoch}_acc_{accuracy:.4f}.pth")
            logger.info(f"New best model saved with val_acc: {accuracy:.4f}, val_loss: {avg_loss:.4f}")

        # Детальные метрики
        if len(all_labels) > 0:
            self._print_metrics(all_labels, all_predictions, all_probs)

        return avg_loss, accuracy

    def _print_metrics(self, labels, predictions, probs):
        """Печать метрик"""
        # Confusion Matrix
        cm = confusion_matrix(labels, predictions)

        # Classification Report
        report = classification_report(labels, predictions,
                                       target_names=['Bird', 'Drone'],
                                       digits=4)

        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"Classification Report:\n{report}")

        # Сохраняем confusion matrix
        self._plot_confusion_matrix(cm, epoch=len(self.history['train_loss']))

    def train(self):
        """Основной цикл обучения"""
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Model: {self.model_type}")
        logger.info(f"Sequence length: {self.config.sequence_length}")
        logger.info(f"Batch size: {self.config.batch_size}")

        for epoch in range(1, self.config.epochs + 1):
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Epoch {epoch}/{self.config.epochs}")
            logger.info(f"{'=' * 50}")

            # Обучение
            train_loss, train_acc = self.train_epoch(epoch)
            logger.info(f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

            # Валидация
            val_loss, val_acc = self.validate(epoch)
            logger.info(f"Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

            # Сохраняем чекпоинт каждые 10 эпох
            if epoch % 10 == 0:
                self.save_model(f"checkpoint_epoch_{epoch}.pth")

            if USE_CLEARML:
                task.get_logger().report_scalar(
                    title="Loss", series="Train", value=train_loss, iteration=epoch
                )
                task.get_logger().report_scalar(
                    title="Accuracy", series="Train", value=train_acc, iteration=epoch
                )
                task.get_logger().report_scalar(
                    title="Loss", series="Val", value=val_loss, iteration=epoch
                )
                task.get_logger().report_scalar(
                    title="Accuracy", series="Val", value=val_acc, iteration=epoch
                )

        # Сохраняем финальную модель
        self.save_model("final_model.pth")

        # Визуализация обучения
        self.plot_training_history()

        logger.info(f"\nTraining completed!")
        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

    def save_model(self, filename):
        """Сохранение модели"""
        save_path = Path(self.config.model_save_dir) / filename
        torch.save({
            'epoch': len(self.history['train_loss']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
            'model_type': self.model_type
        }, save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, path):
        """Загрузка модели"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Model loaded from {path}")

    def plot_training_history(self):
        """Визуализация истории обучения"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train', marker='o')
        axes[0, 0].plot(self.history['val_loss'], label='Val', marker='s')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train', marker='o')
        axes[0, 1].plot(self.history['val_acc'], label='Val', marker='s')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_ylim([0, 1])

        # Learning Rate
        axes[1, 0].plot(self.history['learning_rate'], marker='o', color='green')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale('log')

        # Пустое место для будущих графиков
        axes[1, 1].axis('off')

        plt.tight_layout()
        save_path = Path(self.config.log_dir) / 'training_history.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def _plot_confusion_matrix(self, cm, epoch):
        """Визуализация confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Bird', 'Drone'],
                    yticklabels=['Bird', 'Drone'])
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        save_path = Path(self.config.log_dir) / f'confusion_matrix_epoch_{epoch}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def test_single_sequence(self, video_tensor):
        """Тестирование на одной последовательности"""
        self.model.eval()
        with torch.no_grad():
            video_tensor = video_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
            output = self.model(video_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)

        return predicted.item(), probabilities.cpu().numpy()


# ==================== INFERENCE ====================
class VideoClassifier:
    """Готовый классификатор для использования"""

    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Загружаем модель
        checkpoint = torch.load(model_path, map_location=self.device)
        model_type = checkpoint.get('model_type', 'simple')

        # Создаем модель
        if model_type == 'simple':
            self.model = Simple3DCNN(num_classes=2).to(self.device)
        elif model_type == 'resnet3d':
            self.model = ResNet3D(num_classes=2).to(self.device)
        elif model_type == 'r3d_18':
            self.model = R3D_18(num_classes=2).to(self.device)
        else:
            self.model = Simple3DCNN(num_classes=2).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Трансформации
        self.transform = VideoTransform.get_val_transforms()

        # Конфигурация
        self.sequence_length = checkpoint['config'].get('sequence_length', 16)
        self.img_size = checkpoint['config'].get('img_size', (64, 64))

        logger.info(f"Model loaded: {model_type}")
        logger.info(f"Sequence length: {self.sequence_length}")
        logger.info(f"Image size: {self.img_size}")

    def __call__(self, frames):
        """
        Классификация последовательности кадров.

        Args:
            frames: список изображений (PIL Images или numpy arrays)

        Returns:
            predicted_class: 0 (птица) или 1 (дрон)
            confidence: уверенность предсказания
            probabilities: вероятности для каждого класса
        """
        if len(frames) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} frames, got {len(frames)}")

        # Обрезаем или дополняем до нужной длины
        if len(frames) > self.sequence_length:
            frames = frames[:self.sequence_length]

        # Обработка кадров
        processed_frames = []
        for frame in frames:
            # Конвертируем в PIL если нужно
            if isinstance(frame, np.ndarray):
                # if frame.shape[2] == 3:  # BGR to RGB
                #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame = Image.fromarray(frame)

            # Применяем трансформации
            img_tensor = self.transform(frame)
            processed_frames.append(img_tensor)

        # Создаем видеопоследовательность
        # (sequence_length, C, H, W) -> (C, sequence_length, H, W)
        video_tensor = torch.stack(processed_frames, dim=0)  # (T, C, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # (C, T, H, W)

        # Добавляем batch dimension и перемещаем на устройство
        video_tensor = video_tensor.unsqueeze(0).to(self.device)

        # Предсказание
        with torch.no_grad():
            output = self.model(video_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return {
            'class': predicted.item(),  # 0: bird, 1: drone
            'confidence': confidence.item(),
            'probabilities': probabilities.cpu().numpy()[0],
            'class_names': ['bird', 'drone']
        }


# ==================== TEST FUNCTIONS ====================
def test_data_loading():
    """Тестовая функция для проверки загрузки данных"""
    config = Config()
    # config.data_dir = r"M:\object_crops"  # Укажите правильный путь

    try:
        # Попробуем загрузить датасет
        transform = VideoTransform.get_val_transforms()
        dataset = VideoSequenceDataset(
            root_dir=config.data_dir,
            transform=transform,
            sequence_length=config.sequence_length,
            is_train=False
        )

        print(f"Dataset loaded successfully!")
        print(f"Number of sequences: {len(dataset)}")
        print(f"Number of identities: {len(set(dataset.identities))}")

        # Посмотрим на первый элемент
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample info:")
            print(f"  Video shape: {sample['video'].shape}")  # (C, T, H, W)
            print(f"  Label: {sample['label'].item()}")
            print(f"  Identity: {sample['identity'].item()}")

            # Проверим диапазон значений
            print(f"\nVideo tensor stats:")
            print(f"  Min value: {sample['video'].min().item():.3f}")
            print(f"  Max value: {sample['video'].max().item():.3f}")
            print(f"  Mean value: {sample['video'].mean().item():.3f}")

            # Проверим размеры
            C, T, H, W = sample['video'].shape
            print(f"\nDimensions:")
            print(f"  Channels: {C}")
            print(f"  Time (sequence length): {T}")
            print(f"  Height: {H}")
            print(f"  Width: {W}")

        return True

    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward():
    """Тест forward pass модели"""
    config = Config()

    # Тестируем каждую архитектуру
    model_types = ['simple', 'resnet3d', 'r3d_18']

    for model_type in model_types:
        print(f"\nTesting {model_type}...")

        try:
            trainer = VideoClassifierTrainer(config, model_type)

            # Создаем тестовый тензор
            batch_size = 2
            test_tensor = torch.randn(batch_size, 3, config.sequence_length,
                                      config.img_size[0], config.img_size[1])
            test_tensor = test_tensor.to(config.device)

            # Forward pass
            trainer.model.eval()
            with torch.no_grad():
                output = trainer.model(test_tensor)

            print(f"  Input shape: {test_tensor.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

            # Проверка вероятностей
            probs = torch.softmax(output, dim=1)
            print(
                f"  Probabilities sum to 1: {torch.allclose(probs.sum(dim=1), torch.ones(batch_size).to(config.device), atol=1e-4)}")

        except Exception as e:
            print(f"  Error: {e}")


# ==================== MAIN ====================
def main():
    """Основная функция для обучения"""

    # Сначала протестируем загрузку данных
    print("Testing data loading...")
    if not test_data_loading():
        print("Data loading failed. Please check your data path and structure.")
        return

    # Тестируем forward pass моделей
    print("\nTesting model architectures...")
    test_model_forward()

    # Продолжаем с обучением
    config = Config()
    # config.data_dir = r"M:\object_crops"  # Укажите правильный путь

    print(f"\nStarting training with config:")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Sequence length: {config.sequence_length}")
    print(f"  Image size: {config.img_size}")

    try:
        # Выберите модель: 'simple', 'resnet3d', 'r3d_18'
        model_type = 'simple'  # Начинайте с простой модели

        # Инициализация и обучение
        trainer = VideoClassifierTrainer(config, model_type)
        trainer.train()

        # Тестирование классификатора
        # print("\nTesting classifier...")
        # classifier = VideoClassifier("weights/3d_cnn_classifier/best_model.pth")
        #
        # # Создаем тестовую последовательность
        # test_frames = [np.random.rand(100, 100, 3).astype(np.uint8) for _ in range(config.sequence_length)]
        # result = classifier(test_frames)
        #
        # print(f"Classification result:")
        # print(f"  Class: {result['class_names'][result['class']]} ({result['class']})")
        # print(f"  Confidence: {result['confidence']:.4f}")
        # print(f"  Probabilities: Bird={result['probabilities'][0]:.4f}, Drone={result['probabilities'][1]:.4f}")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()