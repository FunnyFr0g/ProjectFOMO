# encoder_trainer.py
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np


class BirdEncoder(nn.Module):
    """Легковесный энкодер на основе MobileNetV2"""

    def __init__(self, embedding_dim=128, pretrained=True):
        super().__init__()
        # Загружаем предобученную MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=pretrained)

        # Берем все слои кроме последнего классификатора
        self.features = mobilenet.features

        # Адаптивный пулинг для маленьких изображений
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Дополнительные слои для нашего embedding
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

        # Нормализация эмбеддингов для косинусного сходства
        self.norm = nn.functional.normalize

    def forward(self, x):
        # x: [B, C, H, W]
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        embedding = self.fc(pooled)
        return self.norm(embedding, dim=1)  # Нормализованные векторы


class TripletLoss(nn.Module):
    """Triplet loss для обучения энкодера"""

    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


def prepare_triplets(dataset, num_triplets=1000):
    """Создает триплеты для обучения: anchor, positive, negative"""
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    all_embeddings = []
    all_targets = []

    # Сначала получим эмбеддинги для всех изображений
    encoder = BirdEncoder(embedding_dim=128, pretrained=True)
    encoder.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)

    with torch.no_grad():
        for seq, target in loader:
            # Используем целевые кадры как anchor
            target = target.to(device)
            embedding = encoder(target)
            all_embeddings.append(embedding.cpu())
            all_targets.append(0)  # Временная метка

    # Создаем триплеты (упрощенный вариант)
    triplets = []
    # На практике нужно создавать осмысленные триплеты из разных последовательностей
    # Здесь - упрощенная демонстрация

    return triplets


def train_encoder(model, train_loader, val_loader, epochs=20, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    optimizer = Adam(model.parameters(), lr=lr)
    triplet_loss = TripletLoss(margin=0.2)

    # Для классификации как предварительное обучение
    classifier = nn.Linear(128, 2).to(device)  # 2 класса: птица/не птица
    class_optimizer = Adam(list(model.parameters()) + list(classifier.parameters()), lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    os.makedirs('weights/encoder', exist_ok=True)

    for epoch in range(epochs):
        # Training
        model.train()
        classifier.train()
        train_loss = 0

        for batch_idx, (seq, target) in enumerate(train_loader):
            seq, target = seq.to(device), target.to(device)

            # Упрощенное обучение: предсказание "похожести"
            # 1. Получаем эмбеддинги для последовательности и цели
            batch_size, seq_len = seq.shape[:2]

            # Эмбеддинг для целевого кадра (positive)
            target_emb = model(target)

            # Эмбеддинг для последнего кадра последовательности (anchor)
            last_frame = seq[:, -1]  # Последний кадр
            anchor_emb = model(last_frame)

            # Negative: случайный кадр из другой последовательности
            if batch_idx > 0:
                # В реальности нужно брать из другого батча
                neg_idx = torch.randint(0, batch_size, (batch_size,))
                negative = target[neg_idx]
                negative_emb = model(negative)

                # Triplet loss
                loss = triplet_loss(anchor_emb, target_emb, negative_emb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, target in val_loader:
                seq, target = seq.to(device), target.to(device)

                # Проверяем косинусное сходство
                target_emb = model(target)
                last_frame_emb = model(seq[:, -1])

                # Косинусное сходство должно быть высоким
                cosine_sim = F.cosine_similarity(target_emb, last_frame_emb).mean()
                val_loss += (1 - cosine_sim).item()  # Чем меньше, тем лучше

        avg_train = train_loss / len(train_loader) if train_loss > 0 else 0
        avg_val = val_loss / len(val_loader)


        # Сохраняем лучшую модель
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
            }, 'weights/encoder/best.pth')

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train:.4f} | Val Sim Loss: {avg_val:.4f}")

    return model


if __name__ == "__main__":
    from pf_preprocess_sequences import create_datasets

    # Загрузка данных
    train_ds, val_ds = create_datasets("images/vid1/", seq_length=5, img_size=32)
    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=4,           # Параллельная загрузка данных
        pin_memory=True,         # Копирует данные в pinned memory (быстрее на GPU)
        drop_last=True,          # Игнорирует последний неполный батч
        persistent_workers=True  # Workers не пересоздаются каждую эпоху
    )
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # Создание и обучение энкодера
    encoder = BirdEncoder(embedding_dim=128, pretrained=True)
    trained_encoder = train_encoder(
        encoder, train_loader, val_loader,
        epochs=15, lr=1e-4
    )