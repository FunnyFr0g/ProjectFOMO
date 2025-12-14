# evaluate_model_metrics.py
import os
import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Добавляем путь к текущей директории для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Импортируем из существующего файла
from TF_encoder_mobilenet import (
    Config,
    DroneBirdDataset,
    get_transforms,
    SequenceEncoder,
    DataLoader
)
import torch.nn.functional as F


class ModelEvaluator:
    def __init__(self, model_path, config=None, results_dir=None):
        """
        Инициализация оценщика модели

        Args:
            model_path: путь к сохраненной модели (.pth файл)
            config: конфигурация (если None, используется Config по умолчанию)
            results_dir: папка для сохранения результатов (если None, создается автоматически)
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        # Конфигурация
        if config is None:
            self.config = Config()
        else:
            self.config = config

        # Устройство
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Используемое устройство: {self.device}")

        # Папка для результатов
        if results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = Path("evaluation_results") / f"{self.model_path.stem}_{timestamp}"
        else:
            self.results_dir = Path(results_dir)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Результаты будут сохранены в: {self.results_dir}")

        # Загрузка модели и данных
        self.model = None
        self.val_loader = None
        self._load_model()
        self._prepare_data()

    def _load_model(self):
        """Загрузка модели из чекпоинта"""
        logger.info(f"Загрузка модели из {self.model_path}")

        # Создаем модель с теми же параметрами
        self.model = SequenceEncoder(
            embedding_dim=self.config.embedding_dim,
            dropout_rate=self.config.dropout_rate
        ).to(self.device)

        # Загружаем веса
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Пытаемся загрузить напрямую
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        logger.info("Модель успешно загружена")

    def _prepare_data(self):
        """Подготовка валидационного датасета"""
        logger.info("Подготовка валидационного датасета...")

        try:
            # Трансформации для валидации
            val_transform = get_transforms(is_train=False)

            # Создаем валидационный датасет
            val_dataset = DroneBirdDataset(
                root_dir=self.config.data_dir,
                transform=val_transform,
                sequence_length=self.config.sequence_length,
                is_train=False,
                split_ratio=0.8
            )

            if len(val_dataset) == 0:
                logger.error("Валидационный датасет пуст!")
                self.val_loader = None
                return

            # DataLoader
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,  # Устанавливаем 0 для избежания проблем
                pin_memory=True
            )

            # Сохраняем информацию о датасете
            self.val_labels = np.array(val_dataset.labels)
            self.num_samples = len(val_dataset)
            self.num_birds = (self.val_labels == 0).sum()
            self.num_drones = (self.val_labels == 1).sum()

            logger.info(f"Валидационный датасет: {self.num_samples} последовательностей")
            logger.info(f"  - Птицы: {self.num_birds}")
            logger.info(f"  - Дроны: {self.num_drones}")

            # Проверяем наличие обоих классов
            unique_labels = np.unique(self.val_labels)
            if len(unique_labels) < 2:
                logger.warning(f"Только один класс в данных: {unique_labels}")

        except Exception as e:
            logger.error(f"Ошибка при подготовке данных: {e}")
            self.val_loader = None

    def get_predictions(self):
        """Получение предсказаний модели на валидационном датасете"""
        logger.info("Получение предсказаний модели...")

        if self.val_loader is None:
            logger.error("DataLoader не инициализирован!")
            return np.array([]), np.array([]), np.array([])

        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                try:
                    frames = batch['frames'].to(self.device)
                    labels = batch['label'].to(self.device)

                    # Проверяем размер батча
                    if frames.shape[0] == 0:
                        continue

                    # Forward pass
                    _, class_logits = self.model(frames, return_classification=True)

                    # Вероятности
                    probs = F.softmax(class_logits, dim=1)

                    # Предсказанные классы
                    preds = torch.argmax(class_logits, dim=1)

                    # Сохраняем
                    all_labels.append(labels.cpu().numpy())
                    all_preds.append(preds.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())

                    if batch_idx % 10 == 0:
                        logger.info(f"Обработано батчей: {batch_idx + 1}/{len(self.val_loader)}")

                except Exception as e:
                    logger.error(f"Ошибка при обработке батча {batch_idx}: {e}")
                    continue

        # Объединяем все батчи, проверяя что есть данные
        if len(all_labels) == 0:
            logger.error("Нет данных после обработки!")
            return np.array([]), np.array([]), np.array([])

        try:
            y_true = np.concatenate(all_labels)
            y_pred = np.concatenate(all_preds)
            y_proba = np.concatenate(all_probs)

            # Проверяем размерности
            if len(y_true) != len(y_pred) or len(y_true) != y_proba.shape[0]:
                logger.error(
                    f"Размеры не совпадают! y_true={len(y_true)}, y_pred={len(y_pred)}, y_proba={y_proba.shape[0]}")
                # Исправляем - берем минимальный размер
                min_len = min(len(y_true), len(y_pred), y_proba.shape[0])
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]
                y_proba = y_proba[:min_len]

            logger.info(f"Получено предсказаний: {len(y_true)}")
            logger.info(f"Размер y_proba: {y_proba.shape}")

            return y_true, y_pred, y_proba

        except Exception as e:
            logger.error(f"Ошибка при объединении данных: {e}")
            return np.array([]), np.array([]), np.array([])

    def calculate_metrics(self, y_true, y_pred, y_proba):
        """Вычисление всех метрик"""
        logger.info("Вычисление метрик...")

        metrics = {}

        # Проверяем, есть ли данные
        if len(y_true) == 0:
            logger.error("Нет данных для вычисления метрик!")
            metrics['error'] = "No data available"
            return metrics

        # Основные метрики
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # Проверяем наличие обоих классов для бинарных метрик
        unique_labels = np.unique(y_true)
        if len(unique_labels) == 2:
            metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        else:
            logger.warning(f"Обнаружен только один класс: {unique_labels}. Используем 'macro' averaging.")
            metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # ROC-AUC и Precision-Recall AUC
        try:
            # Проверяем, что есть оба класса
            if len(np.unique(y_true)) >= 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                metrics['pr_auc'] = average_precision_score(y_true, y_proba[:, 1])
            else:
                logger.warning("Только один класс в данных, AUC не может быть вычислен")
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
        except Exception as e:
            logger.warning(f"Ошибка при вычислении AUC метрик: {e}")
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0

        # Матрица ошибок
        try:
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()

            # Дополнительные метрики из матрицы ошибок
            if cm.size == 4:  # 2x2 матрица
                tn, fp, fn, tp = cm.ravel()
                metrics['true_positive'] = int(tp)
                metrics['true_negative'] = int(tn)
                metrics['false_positive'] = int(fp)
                metrics['false_negative'] = int(fn)

                metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
                metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
                metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                logger.warning(f"Матрица ошибок имеет размер {cm.shape}, а не 2x2")

        except Exception as e:
            logger.error(f"Ошибка при вычислении матрицы ошибок: {e}")
            metrics['confusion_matrix'] = []

        # Подробный отчет
        try:
            report = classification_report(
                y_true, y_pred,
                target_names=['Bird', 'Drone'],
                output_dict=True,
                zero_division=0
            )
            metrics['classification_report'] = report
        except Exception as e:
            logger.error(f"Ошибка при создании classification report: {e}")
            metrics['classification_report'] = {}

        # Базовая информация
        metrics['num_samples'] = len(y_true)
        metrics['num_birds'] = int((y_true == 0).sum())
        metrics['num_drones'] = int((y_true == 1).sum())

        return metrics

    def plot_roc_curve(self, y_true, y_proba, roc_auc):
        """Построение ROC-кривой"""
        plt.figure(figsize=(10, 8))

        fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1])

        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # Сохраняем
        save_path = self.results_dir / 'roc_curve.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Сохраняем данные - ОБНОВЛЕНО: исправляем размеры массивов
        # Длина thresholds обычно на 1 меньше, чем fpr и tpr
        if len(fpr) == len(tpr):
            if len(thresholds) == len(fpr) - 1:
                # Добавляем последний порог для выравнивания размеров
                thresholds_adjusted = np.append(thresholds, thresholds[-1] if len(thresholds) > 0 else 1.0)
            else:
                # Если размеры уже совпадают
                thresholds_adjusted = thresholds

            # Создаем DataFrame только если все массивы одинаковой длины
            if len(fpr) == len(tpr) == len(thresholds_adjusted):
                roc_data = pd.DataFrame({
                    'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': thresholds_adjusted
                })
                roc_data.to_csv(self.results_dir / 'roc_data.csv', index=False)
            else:
                logger.warning(
                    f"Размеры массивов не совпадают: fpr={len(fpr)}, tpr={len(tpr)}, thresholds={len(thresholds_adjusted)}")
                # Сохраняем отдельно
                roc_data = pd.DataFrame({
                    'fpr': fpr,
                    'tpr': tpr
                })
                roc_data.to_csv(self.results_dir / 'roc_data.csv', index=False)
                # Пороги сохраняем отдельно
                pd.DataFrame({'thresholds': thresholds}).to_csv(
                    self.results_dir / 'roc_thresholds.csv', index=False)
        else:
            logger.warning(f"Размеры fpr и tpr не совпадают: fpr={len(fpr)}, tpr={len(tpr)}")

        logger.info(f"ROC-кривая сохранена: {save_path}")

        return save_path

    def plot_precision_recall_curve(self, y_true, y_proba, pr_auc):
        """Построение Precision-Recall кривой"""
        plt.figure(figsize=(10, 8))

        precision, recall, thresholds = precision_recall_curve(y_true, y_proba[:, 1])

        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'PR curve (AP = {pr_auc:.4f})')

        # Baseline (случайный классификатор)
        positive_ratio = np.mean(y_true)
        plt.axhline(y=positive_ratio, color='r', linestyle='--',
                    label=f'Random (AP = {positive_ratio:.4f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)

        # Сохраняем
        save_path = self.results_dir / 'precision_recall_curve.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Сохраняем данные - ОБНОВЛЕНО: исправляем размеры
        # precision и recall содержат на 1 элемент больше чем thresholds
        if len(precision) == len(recall):
            if len(thresholds) == len(precision) - 1:
                # Создаем DataFrame без последнего элемента precision и recall
                pr_data = pd.DataFrame({
                    'precision': precision[:-1],
                    'recall': recall[:-1],
                    'thresholds': thresholds
                })
            elif len(thresholds) == len(precision):
                # Размеры уже совпадают
                pr_data = pd.DataFrame({
                    'precision': precision,
                    'recall': recall,
                    'thresholds': thresholds
                })
            else:
                # Непредвиденный случай - сохраняем раздельно
                logger.warning(
                    f"Несовпадение размеров: precision={len(precision)}, recall={len(recall)}, thresholds={len(thresholds)}")
                pr_data = pd.DataFrame({
                    'precision': precision,
                    'recall': recall
                })
                pr_data.to_csv(self.results_dir / 'precision_recall_data.csv', index=False)
                pd.DataFrame({'thresholds': thresholds}).to_csv(
                    self.results_dir / 'pr_thresholds.csv', index=False)
                return save_path

            pr_data.to_csv(self.results_dir / 'precision_recall_data.csv', index=False)
        else:
            logger.warning(f"Размеры precision и recall не совпадают: precision={len(precision)}, recall={len(recall)}")

        logger.info(f"Precision-Recall кривая сохранена: {save_path}")

        return save_path

    def plot_confusion_matrix(self, y_true, y_pred, metrics):
        """Построение матрицы ошибок"""
        plt.figure(figsize=(10, 8))

        cm = confusion_matrix(y_true, y_pred)

        # Визуализация
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Bird', 'Drone'],
                    yticklabels=['Bird', 'Drone'])

        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # Сохраняем
        save_path = self.results_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Матрица ошибок сохранена: {save_path}")

        return save_path

    def plot_calibration_curve(self, y_true, y_proba):
        """Построение калибровочной кривой"""
        plt.figure(figsize=(10, 8))

        prob_true, prob_pred = calibration_curve(y_true, y_proba[:, 1], n_bins=10)

        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Classifier')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')

        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Curve (Reliability Diagram)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # Сохраняем
        save_path = self.results_dir / 'calibration_curve.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Калибровочная кривая сохранена: {save_path}")

        return save_path

    def plot_class_distributions(self, y_true, y_proba):
        """Визуализация распределений предсказаний по классам"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Для птиц (класс 0)
        bird_probs = y_proba[y_true == 0, 1]  # Вероятность быть дроном
        if len(bird_probs) > 0:
            axes[0].hist(bird_probs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0].set_title('Prediction Distribution for Birds (True Class 0)')
            axes[0].set_xlabel('Predicted Probability (Drone)')
            axes[0].set_ylabel('Frequency')
            axes[0].grid(True, alpha=0.3)
            axes[0].axvline(0.5, color='red', linestyle='--', label='Decision threshold')
            axes[0].legend()
        else:
            axes[0].text(0.5, 0.5, 'No bird samples', ha='center', va='center')
            axes[0].set_title('No Bird Samples')

        # Для дронов (класс 1)
        drone_probs = y_proba[y_true == 1, 1]  # Вероятность быть дроном
        if len(drone_probs) > 0:
            axes[1].hist(drone_probs, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[1].set_title('Prediction Distribution for Drones (True Class 1)')
            axes[1].set_xlabel('Predicted Probability (Drone)')
            axes[1].set_ylabel('Frequency')
            axes[1].grid(True, alpha=0.3)
            axes[1].axvline(0.5, color='red', linestyle='--', label='Decision threshold')
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, 'No drone samples', ha='center', va='center')
            axes[1].set_title('No Drone Samples')

        plt.tight_layout()

        # Сохраняем
        save_path = self.results_dir / 'class_distributions.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Распределения по классам сохранены: {save_path}")

        return save_path

    def save_metrics(self, metrics, y_true, y_pred, y_proba):
        """Сохранение всех метрик и результатов"""

        # 1. Сохраняем метрики в JSON
        import json

        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_metrics = convert_to_serializable(metrics)

        metrics_path = self.results_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)

        logger.info(f"Метрики сохранены: {metrics_path}")

        # 2. Сохраняем предсказания в CSV
        if len(y_true) > 0:
            predictions_df = pd.DataFrame({
                'true_label': y_true,
                'predicted_label': y_pred,
                'probability_bird': y_proba[:, 0],
                'probability_drone': y_proba[:, 1],
                'correct': (y_true == y_pred).astype(int)
            })

            predictions_path = self.results_dir / 'predictions.csv'
            predictions_df.to_csv(predictions_path, index=False)

            logger.info(f"Предсказания сохранены: {predictions_path}")
        else:
            logger.warning("Нет данных для сохранения предсказаний")

        # 3. Создаем текстовый отчет
        report_path = self.results_dir / 'evaluation_report.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Model: {self.model_path}\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n\n")

            f.write(f"Validation Dataset:\n")
            f.write(f"  Total samples: {metrics.get('num_samples', 0)}\n")
            f.write(f"  Birds (0): {metrics.get('num_birds', 0)}\n")
            f.write(f"  Drones (1): {metrics.get('num_drones', 0)}\n\n")

            f.write("-" * 70 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 70 + "\n\n")

            f.write(f"Accuracy:           {metrics.get('accuracy', 0):.4f}\n")
            f.write(f"Precision:          {metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall:             {metrics.get('recall', 0):.4f}\n")
            f.write(f"F1-Score:           {metrics.get('f1_score', 0):.4f}\n")
            f.write(f"ROC-AUC:            {metrics.get('roc_auc', 0):.4f}\n")
            f.write(f"PR-AUC:             {metrics.get('pr_auc', 0):.4f}\n\n")

            if 'true_positive' in metrics:
                f.write(f"True Positives:     {metrics['true_positive']}\n")
                f.write(f"True Negatives:     {metrics['true_negative']}\n")
                f.write(f"False Positives:    {metrics['false_positive']}\n")
                f.write(f"False Negatives:    {metrics['false_negative']}\n\n")

                f.write(f"True Positive Rate:  {metrics.get('true_positive_rate', 0):.4f}\n")
                f.write(f"True Negative Rate:  {metrics.get('true_negative_rate', 0):.4f}\n")
                f.write(f"False Positive Rate: {metrics.get('false_positive_rate', 0):.4f}\n")
                f.write(f"False Negative Rate: {metrics.get('false_negative_rate', 0):.4f}\n\n")

            f.write("-" * 70 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("-" * 70 + "\n\n")

            report = metrics.get('classification_report', {})
            for class_name in ['Bird', 'Drone', 'macro avg', 'weighted avg']:
                if class_name in report:
                    f.write(f"{class_name}:\n")
                    for metric in ['precision', 'recall', 'f1-score', 'support']:
                        if metric in report[class_name]:
                            f.write(f"  {metric}: {report[class_name][metric]:.4f}\n")
                    f.write("\n")

        logger.info(f"Текстовый отчет сохранен: {report_path}")

        return metrics_path, predictions_path if 'predictions_df' in locals() else None, report_path

    def print_summary(self, metrics):
        """Вывод сводки результатов"""
        print("\n" + "=" * 80)
        print("MODEL EVALUATION SUMMARY")
        print("=" * 80)

        print(f"\n📊 BASIC METRICS:")
        print(f"   Accuracy:          {metrics.get('accuracy', 0):.4f}")
        print(f"   Precision:         {metrics.get('precision', 0):.4f}")
        print(f"   Recall:            {metrics.get('recall', 0):.4f}")
        print(f"   F1-Score:          {metrics.get('f1_score', 0):.4f}")

        print(f"\n🎯 AUC METRICS:")
        print(f"   ROC-AUC:           {metrics.get('roc_auc', 0):.4f}")
        print(f"   PR-AUC:            {metrics.get('pr_auc', 0):.4f}")

        print(f"\n📈 CONFUSION MATRIX:")
        cm = metrics.get('confusion_matrix', [])
        if len(cm) == 2 and len(cm[0]) == 2:
            print(f"   ┌─────────────┬─────────────┐")
            print(f"   │   Predicted │ Bird  Drone │")
            print(f"   ├─────────────┼─────────────┤")
            print(f"   │ Actual Bird │ {cm[0][0]:^5}  {cm[0][1]:^5} │")
            print(f"   │ Actual Drone│ {cm[1][0]:^5}  {cm[1][1]:^5} │")
            print(f"   └─────────────┴─────────────┘")
        else:
            print(f"   Матрица ошибок недоступна или имеет неверный формат")

        print(f"\n📋 DETAILS:")
        report = metrics.get('classification_report', {})
        if 'Bird' in report:
            print(f"   Bird Class - Precision: {report['Bird']['precision']:.4f}, "
                  f"Recall: {report['Bird']['recall']:.4f}, "
                  f"F1: {report['Bird']['f1-score']:.4f}")
        if 'Drone' in report:
            print(f"   Drone Class - Precision: {report['Drone']['precision']:.4f}, "
                  f"Recall: {report['Drone']['recall']:.4f}, "
                  f"F1: {report['Drone']['f1-score']:.4f}")

        print(f"\n💾 All results saved to: {self.results_dir}")
        print("=" * 80)

    def evaluate(self):
        """Основной метод оценки"""
        logger.info("Запуск оценки модели...")

        # Проверяем, что данные загружены
        if self.val_loader is None:
            logger.error("Валидационный датасет не загружен!")
            return {}

        # 1. Получаем предсказания
        y_true, y_pred, y_proba = self.get_predictions()

        # Проверяем, что есть данные
        if len(y_true) == 0:
            logger.error("Нет данных для оценки!")
            return {}

        # 2. Вычисляем метрики
        metrics = self.calculate_metrics(y_true, y_pred, y_proba)

        # 3. Строим графики только если есть данные
        try:
            if 'roc_auc' in metrics and metrics['roc_auc'] > 0:
                self.plot_roc_curve(y_true, y_proba, metrics['roc_auc'])

            if 'pr_auc' in metrics and metrics['pr_auc'] > 0:
                self.plot_precision_recall_curve(y_true, y_proba, metrics['pr_auc'])

            if 'confusion_matrix' in metrics and len(metrics['confusion_matrix']) > 0:
                self.plot_confusion_matrix(y_true, y_pred, metrics)

            self.plot_calibration_curve(y_true, y_proba)
            self.plot_class_distributions(y_true, y_proba)

        except Exception as e:
            logger.error(f"Ошибка при построении графиков: {e}")

        # 4. Сохраняем результаты
        try:
            self.save_metrics(metrics, y_true, y_pred, y_proba)
        except Exception as e:
            logger.error(f"Ошибка при сохранении результатов: {e}")

        # 5. Выводим сводку
        self.print_summary(metrics)

        logger.info("Оценка модели завершена!")

        return metrics


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Оценка классификатора птиц/дронов')

    parser.add_argument('--model_path', type=str, default=r'weights/mobilenet_encoder 32/final_model.pth',
                        help='Путь к сохраненной модели (.pth файл)')

    parser.add_argument('--data_dir', type=str, default=None,
                        help='Путь к данным (если None, используется Config.data_dir)')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Размер батча для оценки')

    parser.add_argument('--results_dir', type=str, default='TF_encoder_evaluation',
                        help='Папка для сохранения результатов')

    args = parser.parse_args()

    try:
        # Загружаем конфигурацию
        config = Config()

        # Обновляем конфигурацию из аргументов
        if args.data_dir:
            config.data_dir = args.data_dir
        if args.batch_size:
            config.batch_size = args.batch_size

        # Создаем оценщик
        evaluator = ModelEvaluator(
            model_path=args.model_path,
            config=config,
            results_dir=args.results_dir
        )

        # Запускаем оценку
        metrics = evaluator.evaluate()

        # Возвращаем ROC-AUC
        print(f"\n🎯 ROC-AUC Score: {metrics.get('roc_auc', 0):.4f}")

        return metrics

    except Exception as e:
        logger.error(f"Ошибка при оценке модели: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()