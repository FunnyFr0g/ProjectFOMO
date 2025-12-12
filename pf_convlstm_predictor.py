# convlstm_predictor.py
import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure

import numpy as np

class ConvLSTMCell(nn.Module):
    """Ячейка ConvLSTM с исправленными размерностями"""

    def __init__(self, input_dim, hidden_dim, kernel_size=3, batch_first=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        padding = kernel_size // 2

        # Исправленный конволюционный слой
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

        # Инициализация весов
        nn.init.orthogonal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x, hidden_state):
        """x: [B, C, H, W] или [T, B, C, H, W] если batch_first=False"""
        if hidden_state is None:
            batch_size = x.size(0) if self.batch_first else x.size(1)
            h_cur = torch.zeros(batch_size, self.hidden_dim, x.size(-2), x.size(-1)).to(x.device)
            c_cur = torch.zeros_like(h_cur)
        else:
            h_cur, c_cur = hidden_state

        # Проверка размерностей
        if x.dim() == 5:  # [B, T, C, H, W] или [T, B, C, H, W]
            raise ValueError("ConvLSTMCell ожидает 4D тензор, получен 5D. "
                             "Используйте последовательный вызов для временных шагов.")

        combined = torch.cat([x, h_cur], dim=1)  # По канальному измерению
        conv_output = self.conv(combined)

        # Разделяем на 4 части
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)

        # Ворота
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Обновление состояния
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTMPredictor(nn.Module):
    """Исправленная модель предсказания с поддержкой последовательностей"""

    def __init__(self, input_channels=3, hidden_dims=[64, 64], output_channels=3):
        super().__init__()

        self.lstm_layers = nn.ModuleList()
        self.hidden_dims = hidden_dims

        # Создаем ячейки ConvLSTM для каждого слоя
        for i, h_dim in enumerate(hidden_dims):
            in_channels = input_channels if i == 0 else hidden_dims[i - 1]
            self.lstm_layers.append(
                ConvLSTMCell(in_channels, h_dim, kernel_size=3, batch_first=True)
            )

        # Декодер
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Адаптивный пулинг для разных размеров
        self.adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))

    def forward(self, x, hidden_states=None):
        """
        Args:
            x: [B, T, C, H, W] - батч последовательностей
            hidden_states: начальные скрытые состояния или None
        Returns:
            pred: [B, C, H, W] - предсказанный кадр
            new_hidden: новые скрытые состояния
        """
        # print(f'{x.shape=}')
        batch_size, seq_len, channels, height, width = x.shape

        # Инициализация скрытых состояний
        if hidden_states is None:
            hidden_states = []
            for h_dim in self.hidden_dims:
                h = torch.zeros(batch_size, h_dim, height, width).to(x.device)
                c = torch.zeros(batch_size, h_dim, height, width).to(x.device)
                hidden_states.append((h, c))

        # Проход по временной последовательности
        for t in range(seq_len):
            current_input = x[:, t]  # [B, C, H, W]

            new_hidden_states = []
            for layer_idx, lstm_cell in enumerate(self.lstm_layers):
                h_cur, c_cur = hidden_states[layer_idx]

                # Если это не первый слой, используем выход предыдущего
                if layer_idx > 0:
                    current_input = new_hidden_states[-1][0]  # h из предыдущего слоя

                # Пропускаем через ячейку ConvLSTM
                h_next, c_next = lstm_cell(current_input, (h_cur, c_cur))
                new_hidden_states.append((h_next, c_next))

            hidden_states = new_hidden_states

        # Берем последнее скрытое состояние из последнего слоя
        last_hidden = hidden_states[-1][0]  # [B, hidden_dim, H, W]

        # Декодируем в изображение
        pred = self.decoder(last_hidden)

        # Адаптируем к исходному размеру если нужно
        if pred.shape[-2:] != (height, width):
            pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=False)

        return pred, hidden_states

    def predict_single_sequence(self, sequence):
        """Предсказание для одной последовательности без батча"""
        # sequence: [T, C, H, W]
        if sequence.dim() == 4:
            sequence = sequence.unsqueeze(0)  # [1, T, C, H, W]
        pred, _ = self.forward(sequence)
        return pred.squeeze(0)  # [C, H, W]


def train_predictor(model, train_loader, val_loader, epochs=50, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    os.makedirs('weights/predictor', exist_ok=True)

    # Комбинированная функция потерь
    def loss_fn(pred, target):
        mse = F.mse_loss(pred, target)
        ssim_val = ssim_metric(pred, target)
        ssim_loss = 1 - ssim_val

        return mse + 0.3 * ssim_loss  # Меньший вес для SSIM

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for seq, target in train_loader:
            seq, target = seq.to(device), target.to(device)

            optimizer.zero_grad()
            pred, _ = model(seq)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, target in val_loader:
                seq, target = seq.to(device), target.to(device)
                pred, _ = model(seq)
                loss = loss_fn(pred, target)
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        scheduler.step(avg_val)

        # Сохраняем лучшую модель
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
            }, 'weights/predictor/best.pth')

        print(
            f"Epoch {epoch + 1}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if epoch % 5 == 0:
            with torch.no_grad():
                # Пример предсказания для визуализации
                test_seq, test_target = next(iter(val_loader))
                test_seq, test_target = test_seq[:1].to(device), test_target[:1].to(device)
                test_pred, _ = model(test_seq)

                # Вычисляем PSNR для мониторинга
                mse = F.mse_loss(test_pred, test_target)
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))

                print(f"Epoch {epoch + 1}/{epochs} | "
                      f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
                      f"PSNR: {psnr:.2f} dB | LR: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    return model


# Дополнительно: функция для визуализации прогресса обучения
def visualize_predictions(model, dataloader, num_samples=1):
    """Визуализация предсказаний для отладки"""
    import matplotlib.pyplot as plt

    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        seq_batch, target_batch = next(iter(dataloader))
        seq_batch, target_batch = seq_batch[:num_samples].to(device), target_batch[:num_samples].to(device)

        pred_batch, _ = model(seq_batch)
        print(f'{pred_batch.shape=}')

        fig, axes = plt.subplots(num_samples, 4, figsize=(12, 3 * num_samples))

        for image in pred_batch:
            image = image.permute(1, 2, 0).numpy()
            cv2.imshow('image', image)
            cv2.waitKey(0)

        for i in range(num_samples):
            # Последний кадр из последовательности
            last_frame = seq_batch[i, -1].cpu().permute(1, 2, 0).numpy()

            # Предсказание
            pred = pred_batch[i].cpu().permute(1, 2, 0).numpy()

            # Целевой кадр
            target = target_batch[i].cpu().permute(1, 2, 0).numpy()

            # Разница
            diff = np.abs(pred - target)

            axes[i, 0].imshow(np.clip(last_frame, 0, 1))
            axes[i, 0].set_title(f"Input (last frame)")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(np.clip(pred, 0, 1))
            axes[i, 1].set_title("Prediction")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(np.clip(target, 0, 1))
            axes[i, 2].set_title("Ground Truth")
            axes[i, 2].axis('off')

            im = axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
            axes[i, 3].set_title("Difference")
            axes[i, 3].axis('off')
            plt.colorbar(im, ax=axes[i, 3])

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    # Загрузка данных
    from pf_preprocess_sequences import create_datasets

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

    # Создание и обучение модели
    model = ConvLSTMPredictor(input_channels=3, hidden_dims=[32, 64])

    # trained_model = train_predictor(
    #     model, train_loader, val_loader,
    #     epochs=30, lr=1e-3
    # )

    checkpoint = torch.load("weights/predictor/best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])


    visualize_predictions(model, val_loader)