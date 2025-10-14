import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

from label_pathes import gt_pathes, pred_pathes


def calculate_iou(box1, box2):
    """Вычисляет Intersection over Union (IoU) между двумя bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    # Площадь пересечения
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Площади bounding boxes
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]

    # Площадь объединения
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def load_coco_json(file_path):
    """Загружает COCO JSON файл"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def organize_annotations_by_image(coco_data):
    """Организует аннотации по изображениям"""
    image_annotations = defaultdict(list)

    # Создаем mapping id -> имя файла
    image_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}

    # Создаем mapping category_id -> имя класса
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    for ann in coco_data['annotations']:
        image_name = image_id_to_name[ann['image_id']]
        class_name = category_id_to_name[ann['category_id']]

        image_annotations[image_name].append({
            'bbox': ann['bbox'],  # [x, y, width, height]
            'category_id': ann['category_id'],
            'class_name': class_name
        })

    return image_annotations, category_id_to_name


def organize_predictions_by_image(coco_data):
    """Организует предсказания по изображениям"""
    image_predictions = defaultdict(list)

    # Создаем mapping id -> имя файла
    image_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}

    # Создаем mapping category_id -> имя класса
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    for ann in coco_data['annotations']:
        image_name = image_id_to_name[ann['image_id']]
        class_name = category_id_to_name[ann['category_id']]

        image_predictions[image_name].append({
            'bbox': ann['bbox'],
            'category_id': ann['category_id'],
            'class_name': class_name,
            'score': ann.get('score', 1.0)  # Если нет score, считаем 1.0
        })

    return image_predictions, category_id_to_name


def calculate_ap_per_class(gt_annotations, pred_annotations, iou_threshold=0.5):
    """Вычисляет AP для каждого класса"""

    # Собираем все предсказания и ground truth по классам
    all_predictions = defaultdict(list)
    all_ground_truth = defaultdict(list)

    # Собираем ground truth
    for image_name, annotations in gt_annotations.items():
        for ann in annotations:
            class_name = ann['class_name']
            all_ground_truth[class_name].append({
                'image_name': image_name,
                'bbox': ann['bbox'],
                'used': False  # Флаг для отслеживания сопоставленных аннотаций
            })

    # Собираем предсказания
    for image_name, predictions in pred_annotations.items():
        for pred in predictions:
            class_name = pred['class_name']
            all_predictions[class_name].append({
                'image_name': image_name,
                'bbox': pred['bbox'],
                'score': pred['score']
            })

    # Вычисляем AP для каждого класса
    ap_results = {}

    for class_name in set(list(all_ground_truth.keys()) + list(all_predictions.keys())):
        gt_list = all_ground_truth.get(class_name, [])
        pred_list = all_predictions.get(class_name, [])

        if not pred_list:
            ap_results[class_name] = 0.0
            continue

        # Сортируем предсказания по confidence score
        pred_list.sort(key=lambda x: x['score'], reverse=True)

        # Сопоставляем предсказания с ground truth
        tp = np.zeros(len(pred_list))
        fp = np.zeros(len(pred_list))

        # Создаем копию ground truth для отслеживания использования
        gt_used = {i: False for i in range(len(gt_list))}

        for i, pred in enumerate(pred_list):
            image_name = pred['image_name']
            pred_bbox = pred['bbox']

            best_iou = 0
            best_gt_idx = -1

            # Ищем соответствующий ground truth в том же изображении
            for j, gt in enumerate(gt_list):
                if gt['image_name'] == image_name and not gt_used[j]:
                    iou = calculate_iou(pred_bbox, gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp[i] = 1
                gt_used[best_gt_idx] = True
            else:
                fp[i] = 1

        # Вычисляем precision-recall кривую
        fp_cumsum = np.cumsum(fp)
        tp_cumsum = np.cumsum(tp)

        recalls = tp_cumsum / len(gt_list) if len(gt_list) > 0 else np.zeros_like(tp_cumsum)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        plt.plot(precisions, recalls)
        plt.show()

        # Вычисляем AP (area under precision-recall curve)
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            mask = recalls >= t
            if mask.any():
                ap += np.max(precisions[mask])

        ap_results[class_name] = ap / 11

    return ap_results


def main():
    # Загрузка данных
    # gt_file = gt_pathes['drones_only_FOMO_val']  # Замените на путь к вашему ground truth файлу
    # pred_file = pred_pathes['drones_only_FOMO_val FOMO_56_104e']  # Замените на путь к вашему predictions файлуданных
    gt_file = gt_pathes['mva23_val']  # Замените на путь к вашему ground truth файлу
    pred_file = pred_pathes['mva23_val YOLO12n 1088px']  # Замените на путь к вашему predictions файлу

    print("Загрузка данных...")
    gt_data = load_coco_json(gt_file)
    pred_data = load_coco_json(pred_file)

    # Организация аннотаций
    print("Организация аннотаций...")
    gt_annotations, gt_categories = organize_annotations_by_image(gt_data)
    pred_annotations, pred_categories = organize_predictions_by_image(pred_data)

    # Различные IoU пороги для тестирования
    iou_thresholds = [0.5,] #[1e-6, 1e-5, 0.0001, 1e-3, 0.01, 0.1]

    results = {}

    print("Вычисление AP для различных IoU порогов...")
    for iou_thresh in iou_thresholds:
        print(f" IoU threshold: {iou_thresh}")
        ap_results = calculate_ap_per_class(gt_annotations, pred_annotations, iou_thresh)
        results[iou_thresh] = ap_results

        # Вывод результатов для текущего порога
        for class_name, ap in ap_results.items():
            print(f"   {class_name}: AP = {ap:.4f}")

        # Средний AP по всем классам
        mean_ap = np.mean(list(ap_results.values()))
        print(f"   mAP: {mean_ap:.4f}\n")

    # Визуализация результатов
    visualize_results(results, iou_thresholds)


def visualize_results(results, iou_thresholds):
    """Визуализирует результаты в виде графика"""

    # Собираем все классы
    all_classes = set()
    for iou_thresh in iou_thresholds:
        all_classes.update(results[iou_thresh].keys())

    # Создаем график
    plt.figure(figsize=(12, 8))

    # Для каждого класса строим кривую AP vs IoU threshold
    for class_name in sorted(all_classes):
        ap_values = []
        for iou_thresh in iou_thresholds:
            ap = results[iou_thresh].get(class_name, 0)
            ap_values.append(ap)

        plt.plot(iou_thresholds, ap_values, marker='o', label=class_name, linewidth=2)

    plt.xlabel('IoU Threshold')
    plt.ylabel('Average Precision (AP)')
    plt.title('AP vs IoU Threshold для различных классов')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(iou_thresholds)
    plt.tight_layout()
    plt.savefig('ap_vs_iou.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Таблица результатов
    print("\n" + "=" * 60)
    print("ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 60)

    # Заголовок таблицы
    header = f"{'Class':<20}" + "".join([f"{f'IoU={iou}':<12}" for iou in iou_thresholds])
    print(header)
    print("-" * (20 + 12 * len(iou_thresholds)))

    # Данные для каждого класса
    for class_name in sorted(all_classes):
        row = f"{class_name:<20}"
        for iou_thresh in iou_thresholds:
            ap = results[iou_thresh].get(class_name, 0)
            row += f"{ap:<12.4f}"
        print(row)

    # Средние значения
    print("-" * (20 + 12 * len(iou_thresholds)))
    mean_row = f"{'mAP':<20}"
    for iou_thresh in iou_thresholds:
        mean_ap = np.mean(list(results[iou_thresh].values()))
        mean_row += f"{mean_ap:<12.4f}"
    print(mean_row)


if __name__ == "__main__":
    main()