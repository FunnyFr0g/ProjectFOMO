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


# def calculate_precision_recall(gt_annotations, pred_annotations, iou_threshold=iou_threshold, size_range=(0, float('inf'))):
#     """
#     Вычисляет PR кривую и AP для детекций объектов с учетом размера GT-боксов
#
#     Параметры:
#     gt_annotations - список аннотаций в формате COCO (истинные метки)
#     pred_annotations - список предсказанных аннотаций в формате COCO
#     iou_threshold - порог IoU для определения true positive
#     size_range - кортеж (min_size, max_size) для фильтрации GT-боксов по размеру
#
#     Возвращает:
#     precision, recall - массивы для построения PR кривой
#     ap - значение Average Precision
#     pos_num - количество положительных примеров в выбранном диапазоне размеров
#     """
#
#     # Собираем все предсказания и истинные боксы по image_id
#     pred_boxes = {}
#     gt_boxes = {}
#
#     # Сначала собираем все GT боксы, чтобы определить, какие из них попадают в нужный размер
#     valid_gt_boxes = set()
#     for gt in gt_annotations:
#         size = max(gt['bbox'][2], gt['bbox'][3])  # Берем максимальный размер (ширину или высоту)
#         if size_range[0] <= size < size_range[1]:
#             valid_gt_boxes.add((gt['image_id'], gt['id']))
#
#     for pred in pred_annotations:
#         image_id = pred['image_id']
#         if image_id not in pred_boxes:
#             pred_boxes[image_id] = []
#         pred_boxes[image_id].append({
#             'bbox': pred['bbox'],
#             'score': pred['score'],
#             'category_id': pred['category_id'],
#             'is_matched': False
#         })
#
#     for gt in gt_annotations:
#         image_id = gt['image_id']
#         if (image_id, gt['id']) not in valid_gt_boxes:
#             continue  # Пропускаем GT боксы, не попадающие в нужный размер
#
#         if image_id not in gt_boxes:
#             gt_boxes[image_id] = []
#         gt_boxes[image_id].append({
#             'bbox': gt['bbox'],
#             'category_id': gt['category_id'],
#             'is_matched': False,
#             'id': gt['id']
#         })
#
#     # Собираем все предсказания и их метки (TP/FP)
#     all_scores = []
#     all_labels = []
#
#     for image_id in pred_boxes:
#         matched_pred_ids = []
#
#         if image_id not in gt_boxes:
#             # Все предсказания на этом изображении - false positives
#             for pred in pred_boxes[image_id]:
#                 all_scores.append(pred['score'])
#                 all_labels.append(0)
#             continue
#
#         image_gt_boxes = gt_boxes[image_id]
#         image_pred_boxes = pred_boxes[image_id]
#
#         # Сортируем предсказания по уверенности
#         image_pred_boxes.sort(key=lambda x: x['score'], reverse=True)
#
#         for pr_id, pred in enumerate(image_pred_boxes):
#             best_iou = 0
#             best_gt_idx = -1
#
#             best_score = 0
#             best_gt_sc_idx = -1
#
#             for gt_idx, gt in enumerate(image_gt_boxes):
#                 if gt['category_id'] != pred['category_id']:
#                     continue
#
#                 if gt['is_matched']:
#                     continue
#
#                 # Вычисляем IoU
#                 iou = calculate_iou(pred['bbox'], gt['bbox'])
#                 if iou > best_iou:
#                     best_iou = iou
#                     best_gt_idx = gt_idx
#
#                 if iou >= iou_threshold:
#                     if pred['score'] > best_score:
#                         best_gt_sc_idx = gt_idx
#
#             matched_pred_ids.append(best_gt_sc_idx)
#             all_scores.append(pred['score'])
#
#         all_labels.extend([1 if i in matched_pred_ids else 0 for i in range(len(image_pred_boxes))])
#
#     pos_num = np.sum(all_labels)
#
#     if pos_num == 0:
#         return [], [], 0.0, 0  # Возвращаем пустые значения, если нет GT-боксов в этом диапазоне
#
#     # Вычисляем Precision-Recall кривую
#     precision, recall, _ = precision_recall_curve(all_labels, all_scores)
#     ap = average_precision_score(all_labels, all_scores)
#
#     return precision, recall, ap


def calculate_precision_recall(gt_annotations, pred_annotations, iou_threshold=0.5):
    """Вычисляет Precision-Recall кривую и AP"""
    pred_boxes = {}
    gt_boxes = {}

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

    for gt in gt_annotations:
        image_id = gt['image_id']
        if image_id not in gt_boxes:
            gt_boxes[image_id] = []
        gt_boxes[image_id].append({
            'bbox': gt['bbox'],
            'category_id': gt['category_id'],
            'matched': False
        })

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

    # Сортируем по убыванию confidence
    indices = np.argsort(-np.array(all_scores))
    all_scores = np.array(all_scores)[indices]
    all_tp = np.array(all_tp)[indices]

    # Вычисляем precision и recall
    tp_cumsum = np.cumsum(all_tp)
    fp_cumsum = np.cumsum(1 - all_tp)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / len([gt for gts in gt_boxes.values() for gt in gts])

    # Добавляем начальную и конечную точки
    precision = np.concatenate([[1], precision, [0]])
    recall = np.concatenate([[0], recall, [1]])

    # Вычисляем AP (Area Under Curve)
    ap = sk_auc(recall, precision)

    return precision, recall, ap


def plot_precision_recall_curve(precision, recall, ap, title, save_path):
    """Визуализирует Precision-Recall кривую"""
    plt.figure()
    plt.plot(recall, precision, label=f'AP = {ap:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title)
    plt.legend(loc="lower left", fontsize=32)
    plt.savefig(save_path)
    plt.close()


def process_all_datasets(gt_pathes, pred_pathes, datasets_list, predict_list, iou_threshold=0.5,
                         output_dir='model_graphics'):
    """Обрабатывает все датасеты и модели"""
    os.makedirs(output_dir, exist_ok=True)

    # Создаем фигуру для всех графиков
    fig, axes = plt.subplots(len(datasets_list), len(predict_list),
                             figsize=(5 * len(predict_list), 5 * len(datasets_list)))
    if len(datasets_list) == 1 or len(predict_list) == 1:
        axes = np.array(axes).reshape(len(datasets_list), len(predict_list))

    ap_results = []

    for i, dataset_name in enumerate(tqdm(datasets_list, desc='Datasets')):
        gt_json = json.load(open(gt_pathes[dataset_name]))
        gt_annotations = gt_json['annotations']

        ds_old_name = dataset_name

        for j, model_name in enumerate(tqdm(predict_list, desc='Models')):

            if 'mva23_val' in dataset_name and "FOMO" in model_name:
                ds_new_name = 'mva23_val_FOMO'
                gt_json = json.load(open(gt_pathes[ds_new_name]))
                gt_annotations = gt_json['annotations']
                pred_name = f"{ds_new_name} {model_name}"
                pred_json = json.load(open(pred_pathes[pred_name]))
            else:
                pred_name = f"{dataset_name} {model_name}"
                pred_json = json.load(open(pred_pathes[pred_name]))

            pred_annotations = pred_json['annotations']

            precision, recall, ap = calculate_precision_recall(
                gt_annotations, pred_annotations, iou_threshold)

            # Сохраняем результаты
            ap_results.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'AP': ap,
                'IoU Threshold': iou_threshold
            })

            # Сохраняем отдельный график
            title = f'{dataset_name} {model_name} IoU={iou_threshold}'
            save_path = os.path.join(output_dir, f'{title}.png')
            plot_precision_recall_curve(precision, recall, ap, title, save_path)

            # Добавляем на общий график
            ax = axes[i, j]
            ax.plot(recall, precision, label=f'AP = {ap:.4f}')
            ax.set_title(title)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_ylim([0.0, 1.05])
            ax.set_xlim([0.0, 1.0])
            ax.legend(loc="lower left", fontsize=32)

            gt_json = json.load(open(gt_pathes[dataset_name]))  # Возвращаем обратно GT_labels
            gt_annotations = gt_json['annotations']

    # Сохраняем общий график
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_curves.png'))
    plt.close()

    # Сохраняем результаты в CSV
    df = pd.DataFrame(ap_results)
    df.to_csv(os.path.join(output_dir, 'ap_results.csv'), index=False)

    return df


# Пример использования
if __name__ == "__main__":
    from label_pathes import gt_pathes, pred_pathes

    # Пример данных (замените на свои)
    # datasets_list = ['mva23_val', 'vid1', 'skb_test']
    # predict_list = ['FOMO 50e', 'FOMO 50e SAHI 800p','YOLO12n 640px','YOLO12n 1088px','baseline',]

    #
    # datasets_list = ['mva23_val_FOMO']
    # predict_list = ['FOMO 50e', 'FOMO 50e SAHI 800p']

    # datasets_list = ['skb_test']
    # predict_list = ['FOMO 50e', 'FOMO 50e no_resize', 'FOMO112 50e', 'FOMO112 10e']

    datasets_list = ['drones_only_FOMO_val', 'drones_only_val']
    model_name_list = ['FOMO_56_104e']
    # datasets_list = ['mva23_val']
    # predict_list = ['YOLO12n 1088px']


    # predict_list = [f'{ds_name} {model_name}' for ds_name in datasets_list for model_name in model_names]
    print(model_name_list)

    # Запускаем обработку
    results = process_all_datasets(
        gt_pathes,
        pred_pathes,
        datasets_list,
        model_name_list,
        iou_threshold
    )

    print("Processing complete. Results saved to model_graphics folder.")