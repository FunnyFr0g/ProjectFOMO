import os
import random
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Конфигурация
class Config:
    # Пути
    data_dir = r"M:\object_crops"  # Измените на свой путь
    model_save_dir = r"weights/mobilenet_encoder 32"
    log_dir = "weights/mobilenet_encoder 32/logs"

    # Параметры данных
    img_size = (32, 32)  # Размер для MobileNet
    sequence_length = 5  # Количество кадров в последовательности
    batch_size = 32
    num_workers = 4

    # Параметры модели
    embedding_dim = 512  # Размерность эмбеддингов
    dropout_rate = 0.3

    # Параметры обучения
    epochs = 50
    learning_rate = 0.001
    weight_decay = 1e-4
    margin = 1.0  # Для triplet loss
    triplet_strategy = "hard"  # "hard", "semi-hard", "all"

    # Sampling
    samples_per_identity = 4  # Количество примеров на один ID в батче

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def create_dirs(cls):
        os.makedirs(cls.model_save_dir, exist_ok=True)
        os.makedirs(cls.log_dir, exist_ok=True)


# Создание директорий
Config.create_dirs()


# ==================== DATASET ====================
class DroneBirdDataset(Dataset):
    def __init__(self, root_dir, transform=None, sequence_length=5, is_train=True, split_ratio=0.8, seed=42):
        """
        Dataset для последовательностей кадров с дронами и птицами.
        Структура данных:
        root_dir/
            DRONE_ID_001/
                frame_001.jpg
                frame_002.jpg
                ...
            BIRD_ID_001/
                frame_001.jpg
                ...
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.sequence_length = sequence_length
        self.is_train = is_train

        # Собираем данные
        self.sequences = []
        self.labels = []  # 0 - птица, 1 - дрон
        self.identities = []  # Уникальные ID для каждого объекта

        self._load_data(split_ratio, seed)
        self._create_sequence_pairs()

        logger.info(f"Загружено {len(self.sequences)} последовательностей")
        logger.info(f"Классы: {np.bincount(self.labels)} (0: птицы, 1: дроны)")
        logger.info(f"Уникальных идентификаторов: {len(set(self.identities))}")

    def _load_data(self, split_ratio, seed):
        """Загрузка и разделение данных"""
        all_sequences = []
        all_labels = []
        all_ids = []

        identity_counter = 0

        # Проходим по всем папкам
        for folder in sorted(self.root_dir.iterdir()):
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

            # Создаем последовательности из этой папки
            for i in range(0, len(images) - self.sequence_length + 1, self.sequence_length // 2):
                seq_images = images[i:i + self.sequence_length]
                if len(seq_images) == self.sequence_length:
                    all_sequences.append(seq_images)
                    all_labels.append(label)
                    all_ids.append(identity_counter)

            identity_counter += 1

        # Разделение на train/val
        if len(all_sequences) > 0:
            train_seqs, val_seqs, train_labels, val_labels, train_ids, val_ids = train_test_split(
                all_sequences, all_labels, all_ids,
                test_size=1 - split_ratio,
                random_state=seed,
                stratify=all_labels
            )
        else:
            train_seqs, val_seqs, train_labels, val_labels, train_ids, val_ids = [], [], [], [], [], []

        if self.is_train:
            self.sequences = train_seqs
            self.labels = train_labels
            self.identities = train_ids
        else:
            self.sequences = val_seqs
            self.labels = val_labels
            self.identities = val_ids

    def _create_sequence_pairs(self):
        """Создаем словарь для быстрого доступа к последовательностям по ID"""
        self.id_to_sequences = defaultdict(list)
        self.id_to_labels = defaultdict(int)

        for seq, label, identity in zip(self.sequences, self.labels, self.identities):
            self.id_to_sequences[identity].append(seq)
            self.id_to_labels[identity] = label

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """Возвращает одну последовательность"""
        seq_paths = self.sequences[idx]
        label = self.labels[idx]
        identity = self.identities[idx]

        # Загружаем и обрабатываем все кадры последовательности
        frames = []
        for img_path in seq_paths:
            try:
                # Загружаем изображение
                img = Image.open(img_path).convert('RGB')

                # Применяем аугментации только при обучении
                if self.transform:
                    if isinstance(self.transform, list):
                        # Разные трансформации для разных кадров
                        img = self.transform[random.randint(0, len(self.transform) - 1)](img)
                    else:
                        img = self.transform(img)

                frames.append(img)
            except Exception as e:
                logger.error(f"Ошибка загрузки {img_path}: {e}")
                # Заполняем нулями в случае ошибки
                frames.append(torch.zeros(3, Config.img_size[0], Config.img_size[1]))

        # Собираем все кадры в один тензор
        if len(frames) > 0:
            frames_tensor = torch.stack(frames)
        else:
            frames_tensor = torch.zeros(self.sequence_length, 3, Config.img_size[0], Config.img_size[1])

        return {
            'frames': frames_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'identity': torch.tensor(identity, dtype=torch.long),
            'sequence_id': idx
        }


# ==================== AUGMENTATIONS ====================
def get_transforms(is_train=True):
    """Аугментации для обучения"""
    if is_train:
        train_transform = transforms.Compose([
            transforms.Resize(Config.img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return train_transform
    else:
        val_transform = transforms.Compose([
            transforms.Resize(Config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return val_transform


# ==================== MODEL ====================
class SequenceEncoder(nn.Module):
    def __init__(self, embedding_dim=512, dropout_rate=0.3):
        super().__init__()

        # Backbone - MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=True)

        # Заменяем классификатор
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

        # Временный модуль для агрегации последовательности
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # Классификатор для птиц/дронов (опционально)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)  # 2 класса: птица, дрон
        )

    def forward(self, x, return_classification=False):
        """
        x: тензор формы (batch_size, sequence_length, 3, H, W)
        """
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Обрабатываем каждый кадр отдельно
        frame_embeddings = []
        for i in range(seq_len):
            # (batch_size, 3, H, W)
            frame = x[:, i, :, :, :]

            # Получаем эмбеддинг для каждого кадра
            embedding = self.backbone(frame)  # (batch_size, embedding_dim)
            frame_embeddings.append(embedding)

        # Собираем все эмбеддинги
        # (batch_size, seq_len, embedding_dim)
        all_embeddings = torch.stack(frame_embeddings, dim=1)

        # Агрегируем по временной оси (берем среднее)
        sequence_embedding = torch.mean(all_embeddings, dim=1)  # (batch_size, embedding_dim)

        # Нормализуем эмбеддинг (важно для triplet loss)
        sequence_embedding = F.normalize(sequence_embedding, p=2, dim=1)

        if return_classification:
            # Предсказание класса (птица/дрон)
            class_logits = self.classifier(sequence_embedding)
            return sequence_embedding, class_logits

        return sequence_embedding

    def extract_features(self, x):
        """Только извлечение признаков без классификации"""
        return self.forward(x, return_classification=False)


# ==================== TRIPLET LOSS ====================
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, strategy="hard"):
        super().__init__()
        self.margin = margin
        self.strategy = strategy

    def forward(self, embeddings, labels):
        """
        embeddings: (batch_size, embedding_dim)
        labels: (batch_size,)
        """
        batch_size = embeddings.size(0)
        device = embeddings.device  # Получаем устройство тензора

        # Создаем torch.arange на том же устройстве
        indices = torch.arange(batch_size, device=device)

        # Вычисляем матрицу расстояний
        dist_matrix = self._pairwise_distance(embeddings)

        # Для каждого анкорного примера находим положительные и отрицательные
        loss = torch.tensor(0.0, device=device)
        num_valid_triplets = 0

        for i in range(batch_size):
            anchor_label = labels[i]

            # Положительные примеры (тот же класс) - на том же устройстве
            positive_mask = (labels == anchor_label) & (indices != i)
            positive_indices = torch.where(positive_mask)[0]

            # Отрицательные примеры (другой класс)
            negative_mask = (labels != anchor_label)
            negative_indices = torch.where(negative_mask)[0]

            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue

            # Выбираем триплеты в зависимости от стратегии
            if self.strategy == "hard":
                # Самый трудный положительный и самый легкий отрицательный
                pos_dist = dist_matrix[i, positive_indices].max()
                neg_dist = dist_matrix[i, negative_indices].min()

                loss = loss + F.relu(pos_dist - neg_dist + self.margin)
                num_valid_triplets += 1

            elif self.strategy == "semi-hard":
                # Все полу-трудные триплеты
                for pos_idx in positive_indices:
                    pos_dist = dist_matrix[i, pos_idx]

                    # Находим отрицательные, которые дальше чем positive + margin
                    hard_negatives = dist_matrix[i, negative_indices] < pos_dist + self.margin

                    if hard_negatives.any():
                        neg_dist = dist_matrix[i, negative_indices][hard_negatives].min()
                        loss = loss + F.relu(pos_dist - neg_dist + self.margin)
                        num_valid_triplets += 1

            elif self.strategy == "all":
                # Все возможные триплеты
                for pos_idx in positive_indices:
                    for neg_idx in negative_indices:
                        pos_dist = dist_matrix[i, pos_idx]
                        neg_dist = dist_matrix[i, neg_idx]

                        loss = loss + F.relu(pos_dist - neg_dist + self.margin)
                        num_valid_triplets += 1

        if num_valid_triplets > 0:
            return loss / num_valid_triplets
        else:
            return torch.tensor(0.0, device=device)

    def _pairwise_distance(self, x):
        """Вычисляет попарные косинусные расстояния"""
        # Нормализуем векторы (уже нормализованы в модели, но на всякий случай)
        x_norm = F.normalize(x, p=2, dim=1)

        # Косинусное сходство
        similarity = torch.mm(x_norm, x_norm.t())

        # Преобразуем в расстояние (1 - similarity)
        distance = 1 - similarity

        # Защита от численных ошибок
        distance = torch.clamp(distance, min=0.0)

        return distance


# ==================== SIMPLIFIED TRIPLET LOSS (альтернатива) ====================
class BatchHardTripletLoss(nn.Module):
    """Упрощенная и более стабильная реализация triplet loss"""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        Batch-hard triplet loss.
        embeddings: (batch_size, embedding_dim)
        labels: (batch_size,)
        """
        # Вычисляем матрицу расстояний
        dist_matrix = self._pairwise_distance(embeddings)

        # Для каждого примера (анкора)
        mask_positive = labels.unsqueeze(0) == labels.unsqueeze(1)  # (batch, batch)
        mask_negative = labels.unsqueeze(0) != labels.unsqueeze(1)  # (batch, batch)

        # Убираем диагональ (расстояние до самого себя)
        eye = torch.eye(dist_matrix.size(0), dtype=torch.bool, device=dist_matrix.device)
        mask_positive = mask_positive & ~eye

        # Проверяем, есть ли положительные и отрицательные примеры для каждого анкора
        valid_positives = mask_positive.sum(dim=1) > 0
        valid_negatives = mask_negative.sum(dim=1) > 0
        valid_mask = valid_positives & valid_negatives

        # Если нет валидных примеров, возвращаем 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=embeddings.device)

        # Работаем только с валидными примерами
        valid_dist_matrix = dist_matrix[valid_mask][:, valid_mask]
        valid_mask_positive = mask_positive[valid_mask][:, valid_mask]
        valid_mask_negative = mask_negative[valid_mask][:, valid_mask]

        # Самый трудный положительный (максимальное расстояние)
        positive_dist = valid_dist_matrix[valid_mask_positive]
        # Решейпим в (n_valid, n_positives_per_valid)
        n_valid = valid_mask.sum().item()
        positive_dist = positive_dist.view(n_valid, -1)

        # Самый трудный отрицательный (минимальное расстояние)
        negative_dist = valid_dist_matrix[valid_mask_negative]
        # Решейпим в (n_valid, n_negatives_per_valid)
        negative_dist = negative_dist.view(n_valid, -1)

        # Проверяем размерности
        if positive_dist.size(1) == 0 or negative_dist.size(1) == 0:
            return torch.tensor(0.0, device=embeddings.device)

        hardest_positive_dist, _ = positive_dist.max(dim=1)
        hardest_negative_dist, _ = negative_dist.min(dim=1)

        # Triplet loss
        losses = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        loss = losses.mean()

        return loss

    def _pairwise_distance(self, x):
        """Вычисляет попарные косинусные расстояния"""
        x_norm = F.normalize(x, p=2, dim=1)
        similarity = torch.mm(x_norm, x_norm.t())
        distance = 1 - similarity
        distance = torch.clamp(distance, min=0.0)
        return distance


# ==================== SIMPLE AND ROBUST TRIPLET LOSS ====================
class RobustTripletLoss(nn.Module):
    """Надежная реализация triplet loss с обработкой краевых случаев"""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        embeddings: (batch_size, embedding_dim)
        labels: (batch_size,) - идентификаторы объектов
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        # Вычисляем матрицу расстояний (косинусное расстояние)
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        similarity = torch.mm(embeddings_norm, embeddings_norm.t())
        distance_matrix = 1 - similarity  # (batch_size, batch_size)
        distance_matrix = torch.clamp(distance_matrix, min=0.0)

        total_loss = torch.tensor(0.0, device=device)
        num_valid = 0

        # Проходим по всем объектам как анкорам
        for i in range(batch_size):
            anchor_label = labels[i]

            # Ищем положительные примеры (тот же ID, но не сам анкор)
            positive_mask = (labels == anchor_label)
            positive_mask[i] = False  # исключаем сам анкор

            # Ищем отрицательные примеры (другой ID)
            negative_mask = (labels != anchor_label)

            positive_indices = torch.where(positive_mask)[0]
            negative_indices = torch.where(negative_mask)[0]

            # Пропускаем, если нет положительных или отрицательных примеров
            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue

            # Находим самый трудный положительный пример (максимальное расстояние)
            hardest_positive_dist = distance_matrix[i, positive_indices].max()

            # Находим самый трудный отрицательный пример (минимальное расстояние)
            hardest_negative_dist = distance_matrix[i, negative_indices].min()

            # Вычисляем triplet loss
            loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)

            total_loss += loss
            num_valid += 1

        # Если нет валидных триплетов, возвращаем 0
        if num_valid == 0:
            return torch.tensor(0.0, device=device)

        return total_loss / num_valid


# ==================== ALTERNATIVE: BATCH ALL TRIPLET LOSS ====================
class BatchAllTripletLoss(nn.Module):
    """Batch-all triplet loss: использует все возможные триплеты в батче"""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        device = embeddings.device

        # Нормализуем эмбеддинги
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)

        # Вычисляем матрицу расстояний
        similarity = torch.mm(embeddings_norm, embeddings_norm.t())
        distances = 1 - similarity
        distances = torch.clamp(distances, min=0.0)

        # Создаем маски для положительных и отрицательных пар
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (batch, batch)

        # Для каждого анкора (i) и позитива (j) находим самый трудный негатив
        losses = []

        for i in range(len(labels)):
            # Положительные примеры для этого анкора
            pos_mask = labels_eq[i].clone()
            pos_mask[i] = False  # исключаем сам анкор

            # Отрицательные примеры
            neg_mask = ~labels_eq[i]

            pos_indices = torch.where(pos_mask)[0]
            neg_indices = torch.where(neg_mask)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            # Для каждого положительного примера
            for j in pos_indices:
                pos_dist = distances[i, j]

                # Находим самый трудный негатив (минимальное расстояние)
                hardest_neg_dist = distances[i, neg_indices].min()

                # Вычисляем loss
                loss = F.relu(pos_dist - hardest_neg_dist + self.margin)
                if loss > 0:
                    losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device)

        return torch.stack(losses).mean()




# ==================== SAMPLER ====================
class TripletSampler:
    """Сэмплер для формирования батчей с триплетами"""

    def __init__(self, dataset, samples_per_identity=4):
        self.dataset = dataset
        self.samples_per_identity = samples_per_identity

        # Группируем индексы по идентификаторам
        self.id_to_indices = defaultdict(list)
        for idx, identity in enumerate(dataset.identities):
            self.id_to_indices[identity].append(idx)

        self.identities = list(self.id_to_indices.keys())

    def __iter__(self):
        # Выбираем случайные идентификаторы для батча
        batch_indices = []

        for _ in range(Config.batch_size // self.samples_per_identity):
            if len(self.identities) == 0:
                break

            # Выбираем случайный ID
            identity = random.choice(self.identities)

            # Выбираем samples_per_identity примеров для этого ID
            indices = self.id_to_indices[identity]
            if len(indices) >= self.samples_per_identity:
                selected = random.sample(indices, self.samples_per_identity)
                batch_indices.extend(selected)

        random.shuffle(batch_indices)
        return iter(batch_indices)

    def __len__(self):
        return len(self.dataset)


# ==================== TRAINING ====================
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device

        # Инициализация модели
        self.model = SequenceEncoder(
            embedding_dim=config.embedding_dim,
            dropout_rate=config.dropout_rate
        ).to(self.device)

        # Функции потерь - используем упрощенную версию
        self.triplet_loss = RobustTripletLoss(margin=config.margin)
        self.classification_loss = nn.CrossEntropyLoss()

        # Оптимизатор
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs
        )

        # Загрузка данных
        self._setup_data()

        # История обучения
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_triplet_loss': [], 'val_triplet_loss': [],
            'train_class_loss': [], 'val_class_loss': [],
            'train_acc': [], 'val_acc': []
        }

        # Лучшая модель
        self.best_val_loss = float('inf')

    def _setup_data(self):
        """Загрузка и подготовка данных"""
        # Трансформации
        train_transform = get_transforms(is_train=True)
        val_transform = get_transforms(is_train=False)

        # Датасеты
        train_dataset = DroneBirdDataset(
            root_dir=self.config.data_dir,
            transform=train_transform,
            sequence_length=self.config.sequence_length,
            is_train=True
        )

        val_dataset = DroneBirdDataset(
            root_dir=self.config.data_dir,
            transform=val_transform,
            sequence_length=self.config.sequence_length,
            is_train=False
        )

        # Проверка, что данные загружены
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            logger.error("Нет данных для обучения. Проверьте путь к данным и структуру папок.")
            raise ValueError("Нет данных для обучения")

        # DataLoader'ы
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,  # Используем shuffle вместо сложного сэмплера
            num_workers=min(self.config.num_workers, 4),  # Ограничиваем для Windows
            pin_memory=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=min(self.config.num_workers, 4),
            pin_memory=True
        )

        logger.info(f"Train: {len(train_dataset)} sequences")
        logger.info(f"Val: {len(val_dataset)} sequences")

        # Информация о классах
        train_labels = np.array(train_dataset.labels)
        val_labels = np.array(val_dataset.labels)
        logger.info(f"Train classes - Birds: {(train_labels == 0).sum()}, Drones: {(train_labels == 1).sum()}")
        logger.info(f"Val classes - Birds: {(val_labels == 0).sum()}, Drones: {(val_labels == 1).sum()}")

    def train_epoch(self, epoch):
        """Одна эпоха обучения"""
        self.model.train()
        total_loss = 0.0
        total_triplet_loss = 0.0
        total_class_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Перемещаем данные на устройство
            frames = batch['frames'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            identities = batch['identity'].to(self.device, non_blocking=True)

            # Forward pass
            self.optimizer.zero_grad()

            # Получаем эмбеддинги и предсказания классов
            embeddings, class_logits = self.model(frames, return_classification=True)

            # Вычисляем потери
            triplet_loss = self.triplet_loss(embeddings, identities)
            class_loss = self.classification_loss(class_logits, labels)

            # Комбинированная функция потерь
            loss = triplet_loss + 0.5 * class_loss

            # Backward pass
            loss.backward()

            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Статистика
            total_loss += loss.item()
            total_triplet_loss += triplet_loss.item()
            total_class_loss += class_loss.item()

            # Точность классификации
            _, predicted = torch.max(class_logits, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Логирование каждые 10 батчей
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | "
                            f"Loss: {loss.item():.4f} (Triplet: {triplet_loss.item():.4f}, "
                            f"Class: {class_loss.item():.4f})")

        # Средние значения за эпоху
        avg_loss = total_loss / len(self.train_loader)
        avg_triplet_loss = total_triplet_loss / len(self.train_loader)
        avg_class_loss = total_class_loss / len(self.train_loader)
        accuracy = total_correct / total_samples

        self.history['train_loss'].append(avg_loss)
        self.history['train_triplet_loss'].append(avg_triplet_loss)
        self.history['train_class_loss'].append(avg_class_loss)
        self.history['train_acc'].append(accuracy)

        return avg_loss, accuracy

    def validate(self, epoch):
        """Валидация"""
        self.model.eval()
        total_loss = 0.0
        total_triplet_loss = 0.0
        total_class_loss = 0.0
        total_correct = 0
        total_samples = 0

        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Перемещаем данные на устройство
                frames = batch['frames'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                identities = batch['identity'].to(self.device, non_blocking=True)

                # Forward pass
                embeddings, class_logits = self.model(frames, return_classification=True)

                # Вычисляем потери
                triplet_loss = self.triplet_loss(embeddings, identities)
                class_loss = self.classification_loss(class_logits, labels)
                loss = triplet_loss + 0.5 * class_loss

                # Статистика
                total_loss += loss.item()
                total_triplet_loss += triplet_loss.item()
                total_class_loss += class_loss.item()

                # Точность классификации
                _, predicted = torch.max(class_logits, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                # Для матрицы ошибок
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Средние значения
        avg_loss = total_loss / max(len(self.val_loader), 1)
        avg_triplet_loss = total_triplet_loss / max(len(self.val_loader), 1)
        avg_class_loss = total_class_loss / max(len(self.val_loader), 1)
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        self.history['val_loss'].append(avg_loss)
        self.history['val_triplet_loss'].append(avg_triplet_loss)
        self.history['val_class_loss'].append(avg_class_loss)
        self.history['val_acc'].append(accuracy)

        # Сохраняем лучшую модель
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_model(f"best_model_epoch_{epoch}.pth")
            logger.info(f"New best model saved with val_loss: {avg_loss:.4f}")

        # Матрица ошибок
        if len(all_labels) > 0:
            cm = confusion_matrix(all_labels, all_predictions)
            logger.info(f"Confusion Matrix:\n{cm}")

        return avg_loss, accuracy

    def train(self):
        """Основной цикл обучения"""
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

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

            # Обновление scheduler
            self.scheduler.step()

            # Сохраняем чекпоинт каждые 5 эпох
            if epoch % 5 == 0:
                self.save_model(f"checkpoint_epoch_{epoch}.pth")

        # Сохраняем финальную модель
        self.save_model("final_model.pth")

        # Визуализация обучения
        self.plot_training_history()

        logger.info("Training completed!")

    def save_model(self, filename):
        """Сохранение модели"""
        save_path = Path(self.config.model_save_dir) / filename
        torch.save({
            'epoch': len(self.history['train_loss']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config.__dict__,
        }, save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, path):
        """Загрузка модели"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")

    def plot_training_history(self):
        """Визуализация истории обучения"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Общие потери
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Triplet loss
        axes[0, 1].plot(self.history['train_triplet_loss'], label='Train')
        axes[0, 1].plot(self.history['val_triplet_loss'], label='Val')
        axes[0, 1].set_title('Triplet Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Classification loss
        axes[0, 2].plot(self.history['train_class_loss'], label='Train')
        axes[0, 2].plot(self.history['val_class_loss'], label='Val')
        axes[0, 2].set_title('Classification Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # Accuracy
        axes[1, 0].plot(self.history['train_acc'], label='Train')
        axes[1, 0].plot(self.history['val_acc'], label='Val')
        axes[1, 0].set_title('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Learning rate
        axes[1, 1].axis('off')

        # Матрица ошибок последней эпохи
        axes[1, 2].axis('off')

        plt.tight_layout()
        save_path = Path(self.config.log_dir) / 'training_history.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def extract_embeddings(self, dataloader):
        """Извлечение эмбеддингов для всего датасета"""
        self.model.eval()
        all_embeddings = []
        all_labels = []
        all_identities = []

        with torch.no_grad():
            for batch in dataloader:
                frames = batch['frames'].to(self.device)
                labels = batch['label']
                identities = batch['identity']

                embeddings = self.model.extract_features(frames)

                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.numpy())
                all_identities.append(identities.numpy())

        if all_embeddings:
            all_embeddings = np.vstack(all_embeddings)
            all_labels = np.concatenate(all_labels)
            all_identities = np.concatenate(all_identities)
        else:
            all_embeddings = np.array([])
            all_labels = np.array([])
            all_identities = np.array([])

        return all_embeddings, all_labels, all_identities


# ==================== INFERENCE ====================
class ReIDEncoder:
    """Готовый энкодер для использования в DeepSORT"""

    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Загружаем модель
        self.model = SequenceEncoder().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Трансформации
        self.transform = transforms.Compose([
            transforms.Resize(Config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, crops):
        """
        Извлекает эмбеддинги для списка кропов.
        crops: список изображений (numpy arrays BGR или PIL Images)
        """
        if not crops:
            return np.array([])

        processed_images = []
        for crop in crops:
            # Конвертируем в PIL если нужно
            if isinstance(crop, np.ndarray):
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = Image.fromarray(crop)

            # Применяем трансформации
            img_tensor = self.transform(crop)
            processed_images.append(img_tensor)

        # Создаем батч с последовательностью длины 1
        batch_tensor = torch.stack(processed_images).unsqueeze(1)  # (N, 1, C, H, W)
        batch_tensor = batch_tensor.to(self.device)

        with torch.no_grad():
            embeddings = self.model.extract_features(batch_tensor)

        # Нормализация (L2)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()


# ==================== TEST FUNCTION ====================
def test_data_loading():
    """Тестовая функция для проверки загрузки данных"""
    config = Config()
    config.data_dir = r"M:\object_crops"  # Укажите правильный путь

    try:
        # Попробуем загрузить датасет
        transform = get_transforms(is_train=False)
        dataset = DroneBirdDataset(
            root_dir=config.data_dir,
            transform=transform,
            sequence_length=config.sequence_length,
            is_train=True
        )

        print(f"Dataset loaded successfully!")
        print(f"Number of sequences: {len(dataset)}")
        print(f"Number of identities: {len(set(dataset.identities))}")

        # Посмотрим на первый элемент
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample info:")
            print(f"  Frames shape: {sample['frames'].shape}")
            print(f"  Label: {sample['label'].item()}")
            print(f"  Identity: {sample['identity'].item()}")

            # Проверим трансформации
            print(f"\nChecking transformations...")
            print(f"  Min value: {sample['frames'].min().item():.3f}")
            print(f"  Max value: {sample['frames'].max().item():.3f}")
            print(f"  Mean value: {sample['frames'].mean().item():.3f}")

        return True

    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==================== MAIN ====================
def main():
    """Основная функция для обучения"""

    # Сначала протестируем загрузку данных
    print("Testing data loading...")
    if not test_data_loading():
        print("Data loading failed. Please check your data path and structure.")
        return

    # Продолжаем с обучением если данные загружены
    config = Config()
    config.data_dir = r"M:\object_crops"  # Укажите правильный путь

    print(f"\nStarting training with config:")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Sequence length: {config.sequence_length}")

    try:
        # Инициализация и обучение
        trainer = Trainer(config)
        trainer.train()

        # Тестирование энкодера
        encoder = ReIDEncoder("saved_models/best_model.pth")

        # Пример использования
        test_images = [np.random.rand(100, 100, 3).astype(np.uint8)] * 3
        embeddings = encoder(test_images)
        print(f"\nEncoder test successful!")
        print(f"  Extracted embeddings shape: {embeddings.shape}")  # (3, 512)

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()