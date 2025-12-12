import json
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
from label_pathes import gt_pathes, pred_pathes


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Вычисляет Intersection over Union (IoU) для двух bounding box'ов

    Args:
        bbox1: [x, y, width, height]
        bbox2: [x, y, width, height]

    Returns:
        IoU значение
    """
    # Конвертируем из [x, y, width, height] в [x1, y1, x2, y2]
    x1_1, y1_1, w1, h1 = bbox1
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1

    x1_2, y1_2, w2, h2 = bbox2
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2

    # Вычисляем площадь пересечения
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Вычисляем площадь объединения
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def calculate_ap(recalls: List[float], precisions: List[float]) -> float:
    """
    Вычисляет Average Precision (AP) по кривой precision-recall

    Args:
        recalls: список recall значений
        precisions: список precision значений

    Returns:
        AP значение
    """
    # Интерполяция precision для 101 точки recall
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]

    # Делаем precision монотонно убывающим
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # Вычисляем AP
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    return ap


def evaluate_detections(
        gt_annotations: List[Dict],
        pred_annotations: List[Dict],
        iou_threshold: float = 0.5,
        num_classes: int = None
) -> Dict:
    """
    Вычисляет метрики детекции для одного IoU threshold

    Args:
        gt_annotations: список Ground Truth аннотаций
        pred_annotations: список предсказаний
        iou_threshold: порог IoU
        num_classes: количество классов

    Returns:
        Словарь с результатами
    """
    # Группируем GT по image_id и category_id
    gt_by_image_class = defaultdict(list)
    for ann in gt_annotations:
        gt_by_image_class[(ann['image_id'], ann['category_id'])].append({
            'bbox': ann['bbox'],
            'used': False  # Флаг для отслеживания сопоставленных GT
        })

    # Сортируем предсказания по confidence score (по убыванию)
    pred_annotations.sort(key=lambda x: x['score'], reverse=True)

    # Словари для хранения результатов по классам
    tp_by_class = defaultdict(list)
    fp_by_class = defaultdict(list)
    scores_by_class = defaultdict(list)
    num_gt_by_class = defaultdict(int)

    # Подсчитываем количество GT для каждого класса
    for ann in gt_annotations:
        num_gt_by_class[ann['category_id']] += 1

    # Обрабатываем каждое предсказание
    for pred in pred_annotations:
        image_id = pred['image_id']
        category_id = pred['category_id']
        pred_bbox = pred['bbox']
        score = pred['score']

        # Получаем GT для этого изображения и класса
        gts = gt_by_image_class.get((image_id, category_id), [])

        best_iou = 0.0
        best_gt_idx = -1

        # Ищем лучший совпадающий GT
        for gt_idx, gt in enumerate(gts):
            if gt['used']:
                continue

            iou = calculate_iou(pred_bbox, gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Определяем, является ли предсказание TP или FP
        is_tp = best_iou >= iou_threshold

        tp_by_class[category_id].append(1 if is_tp else 0)
        fp_by_class[category_id].append(0 if is_tp else 1)
        scores_by_class[category_id].append(score)

        # Помечаем GT как использованный, если нашли совпадение
        if is_tp and best_gt_idx != -1:
            gts[best_gt_idx]['used'] = True

    # Вычисляем метрики для каждого класса
    results = {}
    ap_values = []

    for class_id in sorted(num_gt_by_class.keys()):
        if class_id not in scores_by_class:
            # Нет предсказаний для этого класса
            results[class_id] = {
                'AP': 0.0,
                'precision': [],
                'recall': [],
                'num_gt': num_gt_by_class[class_id],
                'num_pred': 0
            }
            ap_values.append(0.0)
            continue

        tps = np.array(tp_by_class[class_id])
        fps = np.array(fp_by_class[class_id])
        scores = np.array(scores_by_class[class_id])

        # Сортируем по confidence score (уже отсортировано, но на всякий случай)
        sorted_indices = np.argsort(-scores)
        tps = tps[sorted_indices]
        fps = fps[sorted_indices]

        # Вычисляем кумулятивные суммы
        cum_tps = np.cumsum(tps)
        cum_fps = np.cumsum(fps)

        # Вычисляем precision и recall
        precisions = cum_tps / (cum_tps + cum_fps + 1e-10)
        recalls = cum_tps / (num_gt_by_class[class_id] + 1e-10)

        # Вычисляем AP
        ap = calculate_ap(recalls.tolist(), precisions.tolist())

        results[class_id] = {
            'AP': ap,
            'precision': precisions.tolist(),
            'recall': recalls.tolist(),
            'num_gt': num_gt_by_class[class_id],
            'num_pred': len(scores)
        }

        ap_values.append(ap)

    # Вычисляем mAP
    map_value = np.mean(ap_values) if ap_values else 0.0

    return {
        'AP': map_value,
        'AP_per_class': results,
        'iou_threshold': iou_threshold
    }


def evaluate_coco_style(
        gt_file: str,
        pred_file: str,
        output_dir: str = None
) -> Dict:
    """
    Основная функция для оценки в стиле COCO

    Args:
        gt_file: путь к файлу Ground Truth
        pred_file: путь к файлу предсказаний
        output_dir: директория для сохранения результатов

    Returns:
        Словарь с результатами
    """
    # Загружаем данные
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)

    with open(pred_file, 'r') as f:
        pred_data = json.load(f)

    # Получаем аннотации
    gt_annotations = gt_data['annotations']

    # Если pred_data - это словарь с ключом 'annotations', извлекаем его
    if isinstance(pred_data, dict) and 'annotations' in pred_data:
        pred_annotations = pred_data['annotations']
    else:
        pred_annotations = pred_data

    # Вычисляем AP@50
    print("Вычисление AP@50...")
    ap50_results = evaluate_detections(gt_annotations, pred_annotations, iou_threshold=0.5)
    ap50 = ap50_results['AP']

    # Вычисляем mAP@50-95
    print("Вычисление mAP@50-95...")
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    ap_values = []

    for iou_thresh in iou_thresholds:
        results = evaluate_detections(gt_annotations, pred_annotations, iou_threshold=iou_thresh)
        ap_values.append(results['AP'])

    map50_95 = np.mean(ap_values)

    # Выводим результаты
    print("\n" + "=" * 50)
    print("Результаты оценки:")
    print("=" * 50)
    print(f"AP@50: {ap50:.4f}")
    print(f"mAP@50-95: {map50_95:.4f}")

    # Сохраняем результаты
    results = {
        'AP50': float(ap50),
        'mAP50_95': float(map50_95),
        'AP50_per_class': ap50_results['AP_per_class'],
        'iou_thresholds': iou_thresholds.tolist(),
        'AP_per_iou': [float(x) for x in ap_values]
    }

    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nРезультаты сохранены в: {output_path}")

    return results


# Пример использования
if __name__ == "__main__":
    # Пример вызова функции
    results = evaluate_coco_style(
        gt_file=gt_pathes['synth_drone_val'],
        pred_file=pred_pathes['synth_drone_val baseline'],
        output_dir='mAP AP results'
    )