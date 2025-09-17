import json
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO


def analyze_bbox_sizes(annotation_file):
    # Загрузка аннотаций COCO
    coco = COCO(annotation_file)

    # Сбор всех размеров bbox'ов
    widths = []
    heights = []

    # Итерация по всем изображениям и аннотациям
    for img_id in coco.imgs:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        for ann in annotations:
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                widths.append(w)
                heights.append(h)

    # Конвертация в numpy arrays для вычислений
    widths = np.array(widths)
    heights = np.array(heights)

    # Вычисление статистик
    stats = {
        'width': {
            'mean': np.mean(widths),
            'std': np.std(widths),
            'q25': np.percentile(widths, 25),
            'q50': np.percentile(widths, 50),
            'q75': np.percentile(widths, 75),
            'min': np.min(widths),
            'max': np.max(widths)
        },
        'height': {
            'mean': np.mean(heights),
            'std': np.std(heights),
            'q25': np.percentile(heights, 25),
            'q50': np.percentile(heights, 50),
            'q75': np.percentile(heights, 75),
            'min': np.min(heights),
            'max': np.max(heights)
        }
    }

    # Вывод статистик
    print("Статистика по ширине bbox:")
    print(f"  Среднее: {stats['width']['mean']:.2f} ± {stats['width']['std']:.2f} пикселей")
    print(
        f"  Квартили: 25%={stats['width']['q25']:.2f}, 50%={stats['width']['q50']:.2f}, 75%={stats['width']['q75']:.2f}")
    print(f"  Диапазон: [{stats['width']['min']:.2f}, {stats['width']['max']:.2f}]")

    print("\nСтатистика по высоте bbox:")
    print(f"  Среднее: {stats['height']['mean']:.2f} ± {stats['height']['std']:.2f} пикселей")
    print(
        f"  Квартили: 25%={stats['height']['q25']:.2f}, 50%={stats['height']['q50']:.2f}, 75%={stats['height']['q75']:.2f}")
    print(f"  Диапазон: [{stats['height']['min']:.2f}, {stats['height']['max']:.2f}]")

    # Построение гистограмм с квартилями
    plt.figure(figsize=(12, 6))

    # Гистограмма для ширины
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=50, color='blue', alpha=0.7)
    plt.title('Распределение ширины bbox')
    plt.xlabel('Ширина (пиксели)')
    plt.ylabel('Количество')

    # Добавление линий для статистик
    for q, color, label in zip([stats['width']['q25'], stats['width']['q50'], stats['width']['q75']],
                               ['green', 'red', 'green'],
                               ['25%', '50% (медиана)', '75%']):
        plt.axvline(q, color=color, linestyle='dashed', linewidth=1)
        plt.text(q * 1.05, plt.ylim()[1] * 0.85, label, color=color)

    # Гистограмма для высоты
    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=50, color='green', alpha=0.7)
    plt.title('Распределение высоты bbox')
    plt.xlabel('Высота (пиксели)')
    plt.ylabel('Количество')

    # Добавление линий для статистик
    for q, color, label in zip([stats['height']['q25'], stats['height']['q50'], stats['height']['q75']],
                               ['blue', 'red', 'blue'],
                               ['25%', '50% (медиана)', '75%']):
        plt.axvline(q, color=color, linestyle='dashed', linewidth=1)
        plt.text(q * 1.05, plt.ylim()[1] * 0.85, label, color=color)

    plt.tight_layout()
    plt.show()




# Пример использования
if __name__ == "__main__":
    annotation_file = r"X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test\skb_test.json"  # Укажите путь к вашему файлу аннотаций COCO
    analyze_bbox_sizes(annotation_file)