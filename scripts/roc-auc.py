import json
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Загрузка аннотаций


gt_json_path = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\annotations\split_val_coco.json'
pred_json_path = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\annotations\split_val_coco.json'

iou_threshold = 0.25

# Загрузка данных COCO
coco_gt = COCO(gt_json_path)
coco_pred = coco_gt.loadRes(pred_json_path)

# Инициализация COCOeval
coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()


def calculate_iou(box1, box2):
    """
    Вычисляет Intersection over Union (IoU) между двумя bounding box'ами
    box: [x1, y1, w, h] (COCO формат)
    """
    # Преобразуем в формат [x1, y1, x2, y2]
    box1 = [box1['bbox'][0], box1['bbox'][1], box1['bbox'][0] + box1['bbox'][2], box1['bbox'][1] + box1['bbox'][3]]
    box2 = [box2['bbox'][0], box2['bbox'][1], box2['bbox'][0] + box2['bbox'][2], box2['bbox'][1] + box2['bbox'][3]]

    # Определяем координаты пересечения
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Площадь пересечения
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Площадь каждого bounding box'а
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


# Собираем все предсказания и соответствующие метки
image_ids = []
for eval_img in coco_eval.evalImgs:
    if eval_img is None:
        continue
    image_ids.append(eval_img['image_id'])

# Получаем все предсказания и соответствующие метки
scores = []
labels = []

for img_id in image_ids:
    # Получаем предсказания для этого изображения
    pred_anns = coco_pred.loadAnns(coco_pred.getAnnIds(imgIds=[img_id]))
    gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))

    for pred in pred_anns:
        scores.append(pred['score'])
        # Проверяем, есть ли совпадение с ground truth
        ious = [calculate_iou(pred, gt) for gt in gt_anns]  # Нужно реализовать calculate_iou
        labels.append(1 if max(ious) > iou_threshold else 0)  # Порог IoU для TP/FP

# Преобразуем в numpy массивы
scores = np.array(scores)
labels = np.array(labels)

# Вычисляем ROC кривую
fpr, tpr, thresholds = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

# Построение графика
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

print(f'AUC: {roc_auc:.4f}')