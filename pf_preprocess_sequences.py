# preprocess_sequences.py
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
from pathlib import Path
import json


class BirdSequenceDataset(Dataset):
    def __init__(self, data_dir, sequence_length=5, target_size=(32, 32), transform=None):
        """
        Args:
            data_dir: Папка с последовательностями кадров (каждая подпапка - одна траектория)
            sequence_length: Количество кадров для предсказания (N)
            target_size: Размер нормализации (ширина, высота)
            transform: Дополнительные аугментации
        """
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.transform = transform
        self.sequences = []

        # Собираем все последовательности
        data_path = Path(data_dir)
        for track_dir in data_path.iterdir():
            if track_dir.is_dir():
                frames = sorted(track_dir.glob("*.jpg")) + sorted(track_dir.glob("*.png"))
                frames = [str(f) for f in frames]

                # Формируем последовательности: [1,2,3,4,5] -> 6 и т.д.
                for i in range(len(frames) - sequence_length):
                    seq = frames[i:i + sequence_length]
                    target = frames[i + sequence_length]
                    self.sequences.append((seq, target))

        print(f"Создано {len(self.sequences)} последовательностей")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_paths, target_path = self.sequences[idx]

        # Загружаем и обрабатываем последовательность кадров
        sequence = []
        for path in seq_paths:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Не удалось загрузить {path}")

            # Нормализация размера и преобразование цвета
            img = cv2.resize(img, self.target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0  # [0, 1]
            sequence.append(img)

        # Целевой кадр
        target_img = cv2.imread(target_path)
        target_img = cv2.resize(target_img, self.target_size)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        target_img = target_img.astype(np.float32) / 255.0

        # Преобразуем в тензоры (T, C, H, W)
        sequence_tensor = torch.from_numpy(np.stack(sequence)).permute(0, 3, 1, 2)
        target_tensor = torch.from_numpy(target_img).permute(2, 0, 1)

        if self.transform:
            sequence_tensor = self.transform(sequence_tensor)
            target_tensor = self.transform(target_tensor)

        return sequence_tensor, target_tensor


def create_datasets(data_dir, seq_length=5, img_size=32, val_split=0.2):
    """Создает train/val датасеты"""
    from torch.utils.data import random_split

    full_dataset = BirdSequenceDataset(
        data_dir,
        sequence_length=seq_length,
        target_size=(img_size, img_size)
    )

    # Разделение
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    return train_dataset, val_dataset


if __name__ == "__main__":
    # Пример использования
    train_ds, val_ds = create_datasets("images/vid1/", seq_length=5, img_size=32)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Проверка загрузки
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    seq, target = next(iter(loader))
    print(f"Sequence shape: {seq.shape}")  # [4, 5, 3, 32, 32]
    print(f"Target shape: {target.shape}")  # [4, 3, 32, 32]