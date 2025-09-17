import json
import numpy as np
import matplotlib.pyplot as plt


dataset_name = 'vid1' #mva23_val_FOMO'#
model_name = "FOMO 50e" #'YOLO12n 1088px' # # #"YOLO12n 640px" # 'baseline'
predict_name = dataset_name + ' ' + model_name

iou_threshold = 0.1

title = f'{dataset_name} {model_name} IoU={iou_threshold}'

def calculate_roc_auc(gt_annotations, pred_annotations, iou_threshold=iou_threshold):
    """
    Вычисляет ROC кривую и AUC для детекций объектов

    Параметры:
    gt_annotations - список аннотаций в формате COCO (истинные метки)
    pred_annotations - список предсказанных аннотаций в формате COCO
    iou_threshold - порог IoU для определения true positive

    Возвращает:
    fpr, tpr - массивы для построения ROC кривой
    auc - значение площади под кривой
    """

    # Собираем все предсказания и истинные боксы по image_id
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
            'is_matched': False
        })

    for gt in gt_annotations:
        image_id = gt['image_id']
        if image_id not in gt_boxes:
            gt_boxes[image_id] = []
        gt_boxes[image_id].append({
            'bbox': gt['bbox'],
            'category_id': gt['category_id'],
            'is_matched': False
        })

    # Собираем все предсказания и их метки (TP/FP)
    all_scores = []
    all_labels = []

    for image_id in pred_boxes:
        matched_pred_ids = []

        if image_id not in gt_boxes:
            # Все предсказания на этом изображении - false positives
            for pred in pred_boxes[image_id]:
                all_scores.append(pred['score'])
                all_labels.append(0)
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

            # if best_iou >= iou_threshold:
            #     all_labels.append(1)  # TP
            #     image_gt_boxes[best_gt_idx]['is_matched'] = True # нужно смотреть по Score
            # else:
            #     all_labels.append(0)  # FP
            matched_pred_ids.append(best_gt_sc_idx)
            all_scores.append(pred['score'])

        all_labels.extend([1 if i in matched_pred_ids else 0 for i in range(len(image_pred_boxes))])

    pos_num = np.sum(all_labels)
    print(f'Всего {pos_num=} при {iou_threshold=}')





    # Сортируем все предсказания по уверенности
    sorted_indices = np.argsort(-np.array(all_scores))
    all_scores_sorted = np.array(all_scores)[sorted_indices]
    all_labels_sorted = np.array(all_labels)[sorted_indices]

    # Вычисляем TPR и FPR для разных порогов
    total_positives = np.sum(all_labels)
    total_negatives = len(all_labels) - total_positives

    print(f'{total_positives=}')

    tpr = []
    fpr = []

    for threshold in np.linspace(0,1, 50000): # np.unique(all_scores_sorted):
        # Все предсказания с score >= threshold считаем положительными
        tp = np.sum((all_scores_sorted >= threshold) & (all_labels_sorted == 1))
        fp = np.sum((all_scores_sorted >= threshold) & (all_labels_sorted == 0))

        tpr.append(tp / total_positives)
        fpr.append(fp / total_negatives)

    # Добавляем точку (0,0) и (1,1)
    fpr = [0] + fpr + [1]
    tpr = [0] + tpr + [1]

    # сортируем по fpr

    fpr, tpr = zip(*sorted(zip(fpr, tpr)))



    # Вычисляем AUC методом трапеций
    auc = 0
    # for i in range(1, len(fpr)):
    #     auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2

    from scipy.integrate import simpson
    from sklearn.metrics import auc as sk_auc

    # auc = simpson(y=tpr, x=fpr)
    auc = sk_auc(x=fpr,y=tpr)

    return fpr, tpr, auc, pos_num


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


def plot_roc_curve(fpr, tpr, auc, pos_num):
    """Визуализирует ROC кривую"""
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC {title} кол-во TotalPos={pos_num}')
    plt.legend(loc="lower right")
    plt.savefig(r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\ROC' + '\\' + title + '.png')
    plt.show()


# Пример использования
from label_pathes import gt_pathes, pred_pathes

if __name__ == "__main__":


    gt_json = json.load(open(gt_pathes[dataset_name]))
    pred_json = json.load(open(pred_pathes[dataset_name+ ' '+model_name]))

    # gt_json = json.load(open(gt_pathes['skb_test']))
    # pred_json = json.load(open(pred_pathes['skb_test YOLO12n 640px']))

    gt_annotations = gt_json['annotations']
    pred_annotations = pred_json['annotations']

    # # Для примера создадим искусственные данные
    # gt_annotations = [
    #     {'image_id': 1, 'bbox': [10, 10, 20, 20], 'category_id': 1},
    #     {'image_id': 2, 'bbox': [30, 30, 20, 20], 'category_id': 1}
    # ]
    #
    # pred_annotations = [
    #     {'image_id': 1, 'bbox': [12, 12, 18, 18], 'score': 0.9, 'category_id': 1},
    #     {'image_id': 1, 'bbox': [50, 50, 20, 20], 'score': 0.8, 'category_id': 1},
    #     {'image_id': 2, 'bbox': [35, 35, 15, 15], 'score': 0.7, 'category_id': 1},
    #     {'image_id': 2, 'bbox': [10, 10, 20, 20], 'score': 0.6, 'category_id': 1}
    # ]

    fpr, tpr, auc, pos_num = calculate_roc_auc(gt_annotations, pred_annotations)
    print(f"AUC: {auc:.4f}")
    plot_roc_curve(fpr, tpr, auc, pos_num)