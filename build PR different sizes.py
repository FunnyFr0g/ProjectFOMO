import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import precision_recall_curve, average_precision_score

iou_threshold = 1e-3


def calculate_iou(box1, box2):
    """Вычисляет Intersection over Union для двух bounding box'ов"""
    box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection_area / (box1_area + box2_area - intersection_area)


def calculate_precision_recall_with_size_ranges(gt_annotations, pred_annotations, images_info,
                                                iou_threshold=0.5, size_ranges=None):
    """
    Вычисляет Precision-Recall кривую и AP с фильтрацией по размерам объектов

    Параметры:
    gt_annotations - список аннотаций в формате COCO (истинные метки)
    pred_annotations - список предсказанных аннотаций в формате COCO
    images_info - словарь с информацией об изображениях (width, height)
    iou_threshold - порог IoU для определения true positive
    size_ranges - список кортежей [(min_size1, max_size1), ...] для разных диапазонов размеров

    Возвращает:
    results - словарь с результатами для каждого диапазона размеров
    """

    if size_ranges is None:
        size_ranges = [(0, 25), (25, 50), (50, float('inf'))]


    results = {}

    for size_range in size_ranges:
        min_size, max_size = size_range

        pred_boxes = {}
        gt_boxes = {}

        # Собираем GT боксы, попадающие в диапазон размеров
        valid_gt_boxes = set()
        for gt in gt_annotations:
            image_id = gt['image_id']
            if image_id not in images_info:
                continue

            orig_width = images_info[image_id]['width']
            orig_height = images_info[image_id]['height']

            # Вычисляем масштаб для приведения к 224x224
            scale_x = 224.0 / orig_width
            scale_y = 224.0 / orig_height

            # Масштабируем bbox
            bbox = gt['bbox']
            scaled_width = bbox[2] * scale_x
            scaled_height = bbox[3] * scale_y

            # Берем максимальный размер в приведенном разрешении
            scaled_size = max(scaled_width, scaled_height)

            if min_size <= scaled_size < max_size:
                valid_gt_boxes.add((gt['image_id'], gt['id']))

        # Собираем предсказания
        for pred in pred_annotations:
            image_id = pred['image_id']
            if image_id not in pred_boxes:
                pred_boxes[image_id] = []
            pred_boxes[image_id].append({
                'bbox': pred['bbox'],
                'score': pred['score'],
                'category_id': pred['category_id'],
                'matched': False
            })

        # Собираем GT боксы (только valid)
        for gt in gt_annotations:
            image_id = gt['image_id']
            if (image_id, gt['id']) not in valid_gt_boxes:
                continue

            if image_id not in gt_boxes:
                gt_boxes[image_id] = []
            gt_boxes[image_id].append({
                'bbox': gt['bbox'],
                'category_id': gt['category_id'],
                'matched': False
            })

        # Вычисляем precision-recall для текущего диапазона размеров
        all_scores = []
        all_tp = []

        for image_id in pred_boxes:
            if image_id not in gt_boxes:
                for pred in pred_boxes[image_id]:
                    all_scores.append(pred['score'])
                    all_tp.append(0)
                continue

            image_preds = sorted(pred_boxes[image_id], key=lambda x: -x['score'])
            image_gts = gt_boxes[image_id]

            for pred in image_preds:
                best_iou = 0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(image_gts):
                    if gt['matched'] or gt['category_id'] != pred['category_id']:
                        continue

                    iou = calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_threshold:
                    all_tp.append(1)
                    image_gts[best_gt_idx]['matched'] = True
                else:
                    all_tp.append(0)

                all_scores.append(pred['score'])

        # Если нет GT боксов в этом диапазоне
        total_gt = len([gt for gts in gt_boxes.values() for gt in gts])
        if total_gt == 0:
            results[size_range] = {
                'precision': np.array([1, 0]),
                'recall': np.array([0, 1]),
                'ap': 0.0,
                'total_gt': 0
            }
            continue

        # Сортируем по убыванию confidence
        if len(all_scores) > 0:
            indices = np.argsort(-np.array(all_scores))
            all_scores = np.array(all_scores)[indices]
            all_tp = np.array(all_tp)[indices]

            # Вычисляем precision и recall
            tp_cumsum = np.cumsum(all_tp)
            fp_cumsum = np.cumsum(1 - all_tp)

            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / total_gt

            # Добавляем начальную и конечную точки
            precision = np.concatenate([[1], precision, [0]])
            recall = np.concatenate([[0], recall, [1]])

            # Вычисляем AP (Area Under Curve)
            ap = sk_auc(recall, precision)
        else:
            precision = np.array([1, 0])
            recall = np.array([0, 1])
            ap = 0.0

        results[size_range] = {
            'precision': precision,
            'recall': recall,
            'ap': ap,
            'total_gt': total_gt
        }

    return results


def plot_precision_recall_curves_size_ranges(results, title, save_path):
    """Визуализирует Precision-Recall кривые для разных диапазонов размеров"""
    plt.figure(figsize=(10, 8))

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for i, (size_range, result) in enumerate(results.items()):
        min_size, max_size = size_range
        if max_size == float('inf'):
            label = f'Size ≥{min_size}px (AP={result["ap"]:.4f}, GT={result["total_gt"]})'
        else:
            label = f'Size {min_size}-{max_size}px (AP={result["ap"]:.4f}, GT={result["total_gt"]})'

        plt.plot(result['recall'], result['precision'],
                 color=colors[i % len(colors)], label=label, linewidth=2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def process_all_datasets_with_size_ranges(gt_pathes, pred_pathes, datasets_list, predict_list,
                                          iou_threshold=0.5, output_dir='model_graphics_size_ranges'):
    """Обрабатывает все датасеты и модели с анализом по размерам объектов"""
    os.makedirs(output_dir, exist_ok=True)

    # Определяем диапазоны размеров (в пикселях после приведения к 224x224)
    # size_ranges = [(0, 16), (16, 32), (32, 48), (48, 64), (64, float('inf'))]
    size_ranges = [(0, 16), (16, 32), (32, 48), (48, 64), (64, float('inf'))]
    size_ranges = [(4*i, 4*(i+1)) for i in range(7)]
    size_ranges.append((7*4, float('inf')))

    ap_results = []

    for i, dataset_name in enumerate(tqdm(datasets_list, desc='Datasets')):
        gt_json = json.load(open(gt_pathes[dataset_name]))
        gt_annotations = gt_json['annotations']
        images_info = {img['id']: {'width': img['width'], 'height': img['height']}
                       for img in gt_json['images']}

        ds_old_name = dataset_name

        for j, model_name in enumerate(tqdm(predict_list, desc='Models')):
            if 'mva23_val' in dataset_name and "FOMO" in model_name:
                ds_new_name = 'mva23_val_FOMO'
                gt_json = json.load(open(gt_pathes[ds_new_name]))
                gt_annotations = gt_json['annotations']
                images_info = {img['id']: {'width': img['width'], 'height': img['height']}
                               for img in gt_json['images']}
                pred_name = f"{ds_new_name} {model_name}"
                pred_json = json.load(open(pred_pathes[pred_name]))
            else:
                pred_name = f"{dataset_name} {model_name}"
                pred_json = json.load(open(pred_pathes[pred_name]))

            pred_annotations = pred_json['annotations']

            # Вычисляем PR кривые для всех диапазонов размеров
            results = calculate_precision_recall_with_size_ranges(
                gt_annotations, pred_annotations, images_info, iou_threshold, size_ranges)

            # Сохраняем результаты
            for size_range, result in results.items():
                min_size, max_size = size_range
                size_range_str = f"{min_size}-{max_size}" if max_size != float('inf') else f"{min_size}+"

                ap_results.append({
                    'Dataset': dataset_name,
                    'Model': model_name,
                    'Size_Range': size_range_str,
                    'AP': result['ap'],
                    'Total_GT': result['total_gt'],
                    'IoU_Threshold': iou_threshold
                })

            # Сохраняем отдельный график с разными диапазонами размеров
            title = f'{dataset_name} {model_name} IoU={iou_threshold}'
            save_path = os.path.join(output_dir, f'{title}_size_ranges.png')
            plot_precision_recall_curves_size_ranges(results, title, save_path)

            # Возвращаем оригинальные GT annotations для следующей модели
            gt_json = json.load(open(gt_pathes[dataset_name]))
            gt_annotations = gt_json['annotations']
            images_info = {img['id']: {'width': img['width'], 'height': img['height']}
                           for img in gt_json['images']}

    # Сохраняем результаты в CSV
    df = pd.DataFrame(ap_results)
    df.to_csv(os.path.join(output_dir, 'ap_results_size_ranges.csv'), index=False)

    # Создаем сводную таблицу для удобного анализа
    pivot_df = df.pivot_table(index=['Dataset', 'Model'],
                              columns='Size_Range',
                              values='AP',
                              aggfunc='first')
    pivot_df.to_csv(os.path.join(output_dir, 'ap_results_pivot.csv'))

    return df


# Пример использования
if __name__ == "__main__":
    from label_pathes import gt_pathes, pred_pathes

    datasets_list = ['drones_only_FOMO_val', 'drones_only_val']
    model_name_list = ['FOMO_56_104e']

    print(model_name_list)

    # Запускаем обработку с анализом по размерам
    results = process_all_datasets_with_size_ranges(
        gt_pathes,
        pred_pathes,
        datasets_list,
        model_name_list,
        iou_threshold
    )

    print("Processing complete. Results saved to model_graphics_size_ranges folder.")
    print("\nSize ranges analysis:")
    print(" - 0-25px: Very small objects")
    print(" - 25-50px: Small objects")
    print(" - 50-100px: Medium objects")
    print(" - 100+px: Large objects")