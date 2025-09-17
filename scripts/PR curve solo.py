import json
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

from label_pathes import gt_pathes, pred_pathes

# Пути к файлам с аннотациями и предсказаниями



def load_predictions(pred_file, coco_gt=None):
    """Загружает предсказания в формате COCO"""
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)

    # Если это уже COCO-формат (с полями 'images', 'annotations')
    if isinstance(pred_data, dict) and 'images' in pred_data:
        return COCO(pred_file)
    # Если это список детекций (стандартный формат предсказаний)
    elif isinstance(pred_data, list):
        if not coco_gt:
            raise ValueError("Для загрузки предсказаний нужен coco_gt объект")
        return coco_gt.loadRes(pred_data)
    else:
        raise ValueError("Неподдерживаемый формат предсказаний")


def main():
    # Пути к файлам
    gt_file = gt_pathes['mva23_val_FOMO']  # замените на путь к вашему файлу с GT
    pred_file = pred_pathes['mva23_val_FOMO FOMO 50e']  # замените на путь к вашему файлу с предсказаниями
    iou_threshold = 0.01  # COCO или список детекций

    # 1. Загрузка ground truth
    coco_gt = COCO(gt_file)

    # 2. Загрузка предсказаний
    try:
        coco_pred = load_predictions(pred_file, coco_gt)
    except Exception as e:
        print(f"Ошибка загрузки предсказаний: {e}")
        return

    # 3. Инициализация оценки
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')

    # 4. Настройка параметров оценки (опционально)
    coco_eval.params.iouThrs = np.array([iou_threshold])  # IoU threshold = 0.5
    # coco_eval.params.areaRng = [[0, 1e5**2]]    # Все размеры объектов
    # coco_eval.params.maxDets = [100]            # Макс. детекций на изображение

    # 5. Запуск оценки
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 6. Построение PR-кривой
    plot_pr_curve(coco_eval)


def plot_pr_curve(coco_eval, iou_thr=0.1, cat_id=None):
    """Строит PR-кривую для заданного IoU threshold и категории"""
    precision = coco_eval.eval['precision']
    recall = coco_eval.params.recThrs

    # Выбираем precision для:
    # - заданного IoU threshold (по умолчанию 0.5)
    # - всех recall thresholds
    # - заданной категории (или среднее по всем)
    # - area=all
    # - max_dets=100
    iou_idx = np.where(coco_eval.params.iouThrs == iou_thr)[0][0]
    precision = precision[iou_idx, :, :, 0, 2]

    if cat_id is not None:
        cat_idx = coco_eval.params.catIds.index(cat_id)
        precision = precision[:, cat_idx]
    else:
        # Усредняем по всем категориям
        precision = np.mean(precision, axis=1)

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'PR curve (IoU={iou_thr})', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()


if __name__ == '__main__':
    main()