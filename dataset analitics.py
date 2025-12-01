import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from clearml import Dataset


class COCOAnalyzer:
    def __init__(self, annotation_file):
        """
        Инициализация анализатора COCO датасета

        Args:
            annotation_file (str): путь к JSON файлу с аннотациями COCO
        """
        self.annotation_file = annotation_file
        self.coco_data = None
        self.df_annotations = None
        self.df_images = None

        self.load_data()
        self.preprocess_data()

    def load_data(self):
        """Загрузка данных COCO"""
        with open(self.annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        print(f"Загружено {len(self.coco_data['images'])} изображений")
        print(f"Загружено {len(self.coco_data['annotations'])} аннотаций")
        print(f"Загружено {len(self.coco_data['categories'])} категорий")

    def preprocess_data(self):
        """Предобработка данных для анализа"""
        # Создаем DataFrame для изображений
        images_data = []
        for img in self.coco_data['images']:
            images_data.append({
                'image_id': img['id'],
                'width': img['width'],
                'height': img['height'],
                'file_name': img['file_name']
            })
        self.df_images = pd.DataFrame(images_data)

        # Создаем DataFrame для аннотаций
        annotations_data = []
        for ann in self.coco_data['annotations']:
            x, y, w, h = ann['bbox']
            annotations_data.append({
                'image_id': ann['image_id'],
                'category_id': ann['category_id'],
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'area': ann['area'],
                'bbox_area': w * h,
                'aspect_ratio': w / h if h > 0 else 0,
                'center_x': x + w / 2,
                'center_y': y + h / 2
            })

        self.df_annotations = pd.DataFrame(annotations_data)

        # Добавляем информацию о категориях
        category_map = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        self.df_annotations['category_name'] = self.df_annotations['category_id'].map(category_map)

        # Добавляем информацию о размерах изображений к аннотациям
        self.df_annotations = self.df_annotations.merge(
            self.df_images[['image_id', 'width', 'height']],
            on='image_id',
            how='left'
        )

        # Вычисляем относительные размеры и позиции
        self.df_annotations['rel_width'] = self.df_annotations['width_x'] / self.df_annotations['width_y']
        self.df_annotations['rel_height'] = self.df_annotations['height_x'] / self.df_annotations['height_y']
        self.df_annotations['rel_area'] = self.df_annotations['bbox_area'] / (
                    self.df_annotations['width_y'] * self.df_annotations['height_y'])
        self.df_annotations['rel_center_x'] = self.df_annotations['center_x'] / self.df_annotations['width_y']
        self.df_annotations['rel_center_y'] = self.df_annotations['center_y'] / self.df_annotations['height_y']

    def analyze_image_sizes(self):
        """Анализ распределения размеров изображений"""
        print("\n=== АНАЛИЗ РАЗМЕРОВ ИЗОБРАЖЕНИЙ ===")
        print(f"Средняя ширина: {self.df_images['width'].mean():.1f}")
        print(f"Средняя высота: {self.df_images['height'].mean():.1f}")
        print(f"Минимальный размер: {self.df_images[['width', 'height']].min().min()}px")
        print(f"Максимальный размер: {self.df_images[['width', 'height']].max().max()}px")
        print(f"Стандартное отклонение ширины: {self.df_images['width'].std():.1f}")
        print(f"Стандартное отклонение высоты: {self.df_images['height'].std():.1f}")

        # Соотношение сторон изображений
        aspect_ratios = self.df_images['width'] / self.df_images['height']
        print(f"Среднее соотношение сторон (ширина/высота): {aspect_ratios.mean():.2f}")

        return self.df_images[['width', 'height']].describe()

    def analyze_bbox_sizes(self):
        """Анализ распределения размеров bounding box'ов"""
        print("\n=== АНАЛИЗ РАЗМЕРОВ BOUNDING BOX'ОВ ===")
        print(f"Средняя ширина bbox: {self.df_annotations['width_x'].mean():.1f}px")
        print(f"Средняя высота bbox: {self.df_annotations['height_x'].mean():.1f}px")
        print(f"Средняя площадь bbox: {self.df_annotations['bbox_area'].mean():.1f}px²")
        print(f"Минимальная площадь bbox: {self.df_annotations['bbox_area'].min():.1f}px²")
        print(f"Максимальная площадь bbox: {self.df_annotations['bbox_area'].max():.1f}px²")

        # Относительные размеры
        print(f"Средняя относительная ширина: {self.df_annotations['rel_width'].mean():.4f}")
        print(f"Средняя относительная высота: {self.df_annotations['rel_height'].mean():.4f}")
        print(f"Средняя относительная площадь: {self.df_annotations['rel_area'].mean():.6f}")

        return self.df_annotations[
            ['width_x', 'height_x', 'bbox_area', 'rel_width', 'rel_height', 'rel_area']].describe()

    def analyze_bbox_positions(self):
        """Анализ распределения позиций bounding box'ов"""
        print("\n=== АНАЛИЗ ПОЗИЦИЙ BOUNDING BOX'ОВ ===")
        print(f"Средняя позиция центра по X: {self.df_annotations['center_x'].mean():.1f}px")
        print(f"Средняя позиция центра по Y: {self.df_annotations['center_y'].mean():.1f}px")
        print(f"Средняя относительная позиция по X: {self.df_annotations['rel_center_x'].mean():.3f}")
        print(f"Средняя относительная позиция по Y: {self.df_annotations['rel_center_y'].mean():.3f}")

        return self.df_annotations[['center_x', 'center_y', 'rel_center_x', 'rel_center_y']].describe()

    def analyze_aspect_ratios(self):
        """Анализ соотношений сторон"""
        print("\n=== АНАЛИЗ СООТНОШЕНИЙ СТОРОН ===")
        print(f"Среднее соотношение сторон bbox: {self.df_annotations['aspect_ratio'].mean():.2f}")
        print(f"Медианное соотношение сторон bbox: {self.df_annotations['aspect_ratio'].median():.2f}")

        # Классификация bbox по форме
        square_mask = (self.df_annotations['aspect_ratio'] > 0.8) & (self.df_annotations['aspect_ratio'] < 1.2)
        wide_mask = self.df_annotations['aspect_ratio'] >= 1.2
        tall_mask = self.df_annotations['aspect_ratio'] <= 0.8

        print(f"Квадратные bbox: {square_mask.sum()} ({square_mask.mean() * 100:.1f}%)")
        print(f"Широкие bbox: {wide_mask.sum()} ({wide_mask.mean() * 100:.1f}%)")
        print(f"Высокие bbox: {tall_mask.sum()} ({tall_mask.mean() * 100:.1f}%)")

    def analyze_by_category(self):
        """Анализ распределения по категориям"""
        print("\n=== АНАЛИЗ ПО КАТЕГОРИЯМ ===")
        category_stats = self.df_annotations.groupby('category_name').agg({
            'bbox_area': ['count', 'mean', 'std', 'min', 'max'],
            'aspect_ratio': ['mean', 'std'],
            'rel_area': ['mean', 'std']
        }).round(2)

        print(category_stats)
        return category_stats

    def create_visualizations(self, output_dir='coco_analysis'):
        """Создание визуализаций для анализа"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Настройка стиля графиков
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Распределение размеров изображений
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Размеры изображений
        axes[0, 0].hist(self.df_images['width'], bins=50, alpha=0.7)
        axes[0, 0].set_title('Распределение ширины изображений')
        axes[0, 0].set_xlabel('Ширина (px)')
        axes[0, 0].set_ylabel('Количество')

        axes[0, 1].hist(self.df_images['height'], bins=50, alpha=0.7, color='orange')
        axes[0, 1].set_title('Распределение высоты изображений')
        axes[0, 1].set_xlabel('Высота (px)')
        axes[0, 1].set_ylabel('Количество')

        # Соотношение сторон изображений
        aspect_ratios_img = self.df_images['width'] / self.df_images['height']
        axes[1, 0].hist(aspect_ratios_img, bins=50, alpha=0.7, color='green')
        axes[1, 0].set_title('Распределение соотношений сторон изображений')
        axes[1, 0].set_xlabel('Соотношение сторон (ширина/высота)')
        axes[1, 0].set_ylabel('Количество')

        # Scatter plot размеров изображений
        axes[1, 1].scatter(self.df_images['width'], self.df_images['height'], alpha=0.5)
        axes[1, 1].set_title('Размеры изображений')
        axes[1, 1].set_xlabel('Ширина (px)')
        axes[1, 1].set_ylabel('Высота (px)')

        plt.tight_layout()
        plt.savefig(output_path / 'image_sizes_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Распределение размеров bbox
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Абсолютные размеры bbox
        axes[0, 0].hist(self.df_annotations['width_x'], bins=50, alpha=0.7)
        axes[0, 0].set_title('Распределение ширины bbox')
        axes[0, 0].set_xlabel('Ширина bbox (px)')
        axes[0, 0].set_ylabel('Количество')

        axes[0, 1].hist(self.df_annotations['height_x'], bins=50, alpha=0.7, color='orange')
        axes[0, 1].set_title('Распределение высоты bbox')
        axes[0, 1].set_xlabel('Высота bbox (px)')
        axes[0, 1].set_ylabel('Количество')

        axes[0, 2].hist(self.df_annotations['bbox_area'], bins=50, alpha=0.7, color='red')
        axes[0, 2].set_title('Распределение площади bbox')
        axes[0, 2].set_xlabel('Площадь bbox (px²)')
        axes[0, 2].set_ylabel('Количество')

        # Относительные размеры bbox
        axes[1, 0].hist(self.df_annotations['rel_width'], bins=50, alpha=0.7)
        axes[1, 0].set_title('Относительная ширина bbox')
        axes[1, 0].set_xlabel('Ширина bbox / Ширина изображения')
        axes[1, 0].set_ylabel('Количество')

        axes[1, 1].hist(self.df_annotations['rel_height'], bins=50, alpha=0.7, color='orange')
        axes[1, 1].set_title('Относительная высота bbox')
        axes[1, 1].set_xlabel('Высота bbox / Высота изображения')
        axes[1, 1].set_ylabel('Количество')

        axes[1, 2].hist(self.df_annotations['rel_area'], bins=50, alpha=0.7, color='red')
        axes[1, 2].set_title('Относительная площадь bbox')
        axes[1, 2].set_xlabel('Площадь bbox / Площадь изображения')
        axes[1, 2].set_ylabel('Количество')

        plt.tight_layout()
        plt.savefig(output_path / 'bbox_sizes_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Позиции и соотношения сторон bbox
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Позиции центров bbox
        axes[0, 0].hist2d(self.df_annotations['rel_center_x'], self.df_annotations['rel_center_y'],
                          bins=50, cmap='Blues')
        axes[0, 0].set_title('Распределение центров bbox')
        axes[0, 0].set_xlabel('Относительная позиция X')
        axes[0, 0].set_ylabel('Относительная позиция Y')

        # Соотношения сторон bbox
        axes[0, 1].hist(self.df_annotations['aspect_ratio'], bins=50, alpha=0.7, color='purple')
        axes[0, 1].set_title('Распределение соотношений сторон bbox')
        axes[0, 1].set_xlabel('Соотношение сторон (ширина/высота)')
        axes[0, 1].set_ylabel('Количество')
        axes[0, 1].set_xlim(0, 5)  # Ограничиваем для лучшей визуализации

        # Размеры bbox в 2D
        axes[1, 0].scatter(self.df_annotations['width_x'], self.df_annotations['height_x'],
                           alpha=0.3, s=1)
        axes[1, 0].set_title('Размеры bbox (ширина vs высота)')
        axes[1, 0].set_xlabel('Ширина bbox (px)')
        axes[1, 0].set_ylabel('Высота bbox (px)')

        # Количество bbox по категориям
        category_counts = self.df_annotations['category_name'].value_counts()
        axes[1, 1].barh(range(len(category_counts)), category_counts.values)
        axes[1, 1].set_yticks(range(len(category_counts)))
        axes[1, 1].set_yticklabels(category_counts.index)
        axes[1, 1].set_title('Количество bbox по категориям')
        axes[1, 1].set_xlabel('Количество bbox')

        plt.tight_layout()
        plt.savefig(output_path / 'bbox_positions_aspect_ratios.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nВизуализации сохранены в директорию: {output_dir}")

    def generate_report(self, output_file='coco_analysis_report.txt'):
        """Генерация текстового отчета"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ ПО АНАЛИЗУ COCO ДАТАСЕТА\n")
            f.write("=" * 50 + "\n\n")

            f.write("ОБЩАЯ СТАТИСТИКА:\n")
            f.write(f"Количество изображений: {len(self.df_images)}\n")
            f.write(f"Количество аннотаций: {len(self.df_annotations)}\n")
            f.write(f"Количество категорий: {len(self.coco_data['categories'])}\n")
            f.write(f"Среднее количество bbox на изображение: {len(self.df_annotations) / len(self.df_images):.2f}\n\n")

            f.write("РАЗМЕРЫ ИЗОБРАЖЕНИЙ:\n")
            f.write(self.analyze_image_sizes().to_string() + "\n\n")

            f.write("РАЗМЕРЫ BOUNDING BOX'ОВ:\n")
            f.write(self.analyze_bbox_sizes().to_string() + "\n\n")

            f.write("ПОЗИЦИИ BOUNDING BOX'ОВ:\n")
            f.write(self.analyze_bbox_positions().to_string() + "\n\n")

            f.write("СТАТИСТИКА ПО КАТЕГОРИЯМ:\n")
            f.write(self.analyze_by_category().to_string() + "\n")

        print(f"Текстовый отчет сохранен в: {output_file}")


def main():
    # Пример использования
    annotation_file = "GTlabels/ds_45062c8b1fac490480d105ad9c945f22/train/train_annotations.json"  # Укажите путь к вашему файлу

    # Инициализация анализатора
    analyzer = COCOAnalyzer(annotation_file)

    # Запуск анализа
    analyzer.analyze_image_sizes()
    analyzer.analyze_bbox_sizes()
    analyzer.analyze_bbox_positions()
    analyzer.analyze_aspect_ratios()
    analyzer.analyze_by_category()

    # Создание визуализаций
    analyzer.create_visualizations()

    # Генерация отчета
    analyzer.generate_report()


if __name__ == "__main__":
    main()