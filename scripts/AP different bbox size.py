import json
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_recall_curve, average_precision_score

dataset_name = 'skb_test'  # 'mva23_val_FOMO'#
model_name = "FOMO 50e"  # 'YOLO12n 1088px' # # #"YOLO12n 640px" # 'baseline'
predict_name = dataset_name + ' ' + model_name

iou_threshold = 0.01
size_bins = [i for i in range(0,101,25)]  # Интервалы размеров GT-боксов

# Создаем папку для сохранения графиков
output_dir = r'PR_size_bins'
os.makedirs(output_dir, exist_ok=True)


def calculate_pr_ap(gt_annotations, pred_annotations, iou_threshold=iou_threshold, size_range=(0, float('inf'))):
    """
    Вычисляет PR кривую и AP для детекций объектов с учетом размера GT-боксов

    Параметры:
    gt_annotations - список аннотаций в формате COCO (истинные метки)
    pred_annotations - список предсказанных аннотаций в формате COCO
    iou_threshold - порог IoU для определения true positive
    size_range - кортеж (min_size, max_size) для фильтрации GT-боксов по размеру

    Возвращает:
    precision, recall - массивы для построения PR кривой
    ap - значение Average Precision
    pos_num - количество положительных примеров в выбранном диапазоне размеров
    """

    # Собираем все предсказания и истинные боксы по image_id
    pred_boxes = {}
    gt_boxes = {}

    # Сначала собираем все GT боксы, чтобы определить, какие из них попадают в нужный размер
    valid_gt_boxes = set()
    for gt in gt_annotations:
        size = max(gt['bbox'][2], gt['bbox'][3])  # Берем максимальный размер (ширину или высоту)
        if size_range[0] <= size < size_range[1]:
            valid_gt_boxes.add((gt['image_id'], gt['id']))

    for pred in pred_annotations:
        image_id = pred['image_id']
        if image_id not in pred_boxes:
            pred_boxes[image_id] = []
        pred_boxes[image_id].append({
            'bbox': pred['bbox'],
            'score': pred['score'],
            'category_id': pred['category_id'],
            'is_matched': False
        })

    for gt in gt_annotations:
        image_id = gt['image_id']
        if (image_id, gt['id']) not in valid_gt_boxes:
            continue  # Пропускаем GT боксы, не попадающие в нужный размер

        if image_id not in gt_boxes:
            gt_boxes[image_id] = []
        gt_boxes[image_id].append({
            'bbox': gt['bbox'],
            'category_id': gt['category_id'],
            'is_matched': False,
            'id': gt['id']
        })

    # Собираем все предсказания и их метки (TP/FP)
    all_scores = []
    all_labels = []

    for image_id in pred_boxes:
        matched_pred_ids = []

        if image_id not in gt_boxes:
            # Все предсказания на этом изображении - false positives
            # for pred in pred_boxes[image_id]:
            #     all_scores.append(pred['score'])
            #     all_labels.append(0)
            continue

        image_gt_boxes = gt_boxes[image_id]
        image_pred_boxes = pred_boxes[image_id]

        # Сортируем предсказания по уверенности
        image_pred_boxes.sort(key=lambda x: x['score'], reverse=True)

        for pr_id, pred in enumerate(image_pred_boxes):
            best_iou = 0
            best_gt_idx = -1

            best_score = 0
            best_gt_sc_idx = -1

            for gt_idx, gt in enumerate(image_gt_boxes):
                if gt['category_id'] != pred['category_id']:
                    continue

                if gt['is_matched']:
                    continue

                # Вычисляем IoU
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

                if iou >= iou_threshold:
                    if pred['score'] > best_score:
                        best_gt_sc_idx = gt_idx

            matched_pred_ids.append(best_gt_sc_idx)
            all_scores.append(pred['score'])

        all_labels.extend([1 if i in matched_pred_ids else 0 for i in range(len(image_pred_boxes))])

    pos_num = np.sum(all_labels)

    if pos_num == 0:
        return [], [], 0.0, 0  # Возвращаем пустые значения, если нет GT-боксов в этом диапазоне

    # Вычисляем Precision-Recall кривую
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)

    return precision, recall, ap, pos_num


def calculate_iou(box1, box2):
    """
    Вычисляет Intersection over Union для двух bounding box'ов

    Параметры:
    box1, box2 - [x, y, width, height]

    Возвращает:
    iou - значение от 0 до 1
    """
    # Конвертируем в формат [x1, y1, x2, y2]
    box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    # Находим координаты пересечения
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / (box1_area + box2_area - intersection_area)
    return iou


def plot_pr_curve(precision, recall, ap, pos_num, size_range, output_dir):
    """Визуализирует PR кривую для заданного диапазона размеров"""
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {ap:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    # Формируем заголовок и имя файла
    size_str = f"{size_range[0]}-{size_range[1]}"
    title = f'{dataset_name} {model_name} IoU={iou_threshold} Size={size_str}'
    plt.title(f'PR {title}\nTotalPos={pos_num}')
    plt.legend(loc="lower left", fontsize=24)

    # Сохраняем в файл
    filename = f"PR_{dataset_name}_{model_name}_IoU{iou_threshold}_Size{size_str}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved PR curve to {output_path}")


if __name__ == "__main__":
    from label_pathes import gt_pathes, pred_pathes

    gt_json = json.load(open(gt_pathes[dataset_name]))
    pred_json = json.load(open(pred_pathes[dataset_name + ' ' + model_name]))

    gt_annotations = gt_json['annotations']
    pred_annotations = pred_json['annotations']

    # Строим PR-кривые для каждого интервала размеров
    for i in range(len(size_bins) - 1):
        min_size = size_bins[i]
        max_size = size_bins[i + 1]

        print(f"\nCalculating PR for size range {min_size}-{max_size}")
        precision, recall, ap, pos_num = calculate_pr_ap(
            gt_annotations,
            pred_annotations,
            iou_threshold=iou_threshold,
            size_range=(min_size, max_size)
        )

        print(f"AP for size {min_size}-{max_size}: {ap:.4f}, TotalPos={pos_num}")

        if pos_num > 0:  # Строим график только если есть GT-боксы в этом диапазоне
            plot_pr_curve(precision, recall, ap, pos_num, (min_size, max_size), output_dir)
        else:
            print(f"No GT boxes in size range {min_size}-{max_size}, skipping...")

    # Также строим общую PR-кривую для всех размеров
    print("\nCalculating PR for all sizes")
    precision, recall, ap, pos_num = calculate_pr_ap(
        gt_annotations,
        pred_annotations,
        iou_threshold=iou_threshold,
        size_range=(0, float('inf'))
    )
    print(f"AP for all sizes: {ap:.4f}, TotalPos={pos_num}")
    plot_pr_curve(precision, recall, ap, pos_num, (0, float('inf')), output_dir)