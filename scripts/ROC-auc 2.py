import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
from label_pathes import gt_pathes, pred_pathes

# dataset_name = 'vid1'
# model_name = 'YOLO12n 1088px2' #"YOLO12n 640px" # "baseline"

dataset_name = 'mva23_val_FOMO'
model_name = "FOMO 50e" #'YOLO12n 1088px' # # #"YOLO12n 640px" # 'baseline'
predict_name = dataset_name + ' ' + model_name

iou_threshold = 0.001







def load_and_validate_coco(gt_path, pred_path):
    """Загружает и валидирует COCO аннотации и предсказания"""
    try:
        # Загрузка ground truth
        coco_gt = COCO(gt_path)

        # Проверка формата предсказаний
        with open(pred_path) as f:
            pred_data = json.load(f)

        # Если предсказания в формате COCO results (список)
        if isinstance(pred_data, list):
            # Проверяем обязательные поля
            required_fields = ['image_id', 'category_id', 'bbox', 'score']
            if not all(field in pred_data[0] for field in required_fields):
                raise ValueError("Предсказания должны содержать поля: image_id, category_id, bbox, score")

            # Сохраняем временный файл с предсказаниями
            temp_pred_path = 'temp_predictions.json'
            with open(temp_pred_path, 'w') as f:
                json.dump(pred_data, f)
            return coco_gt, temp_pred_path

        # Если предсказания в формате COCO output (словарь)
        elif isinstance(pred_data, dict) and 'annotations' in pred_data:
            temp_pred_path = 'temp_predictions.json'
            with open(temp_pred_path, 'w') as f:
                json.dump(pred_data['annotations'], f)
            return coco_gt, temp_pred_path

        else:
            raise ValueError("Неподдерживаемый формат предсказаний")

    except Exception as e:
        raise ValueError(f"Ошибка загрузки данных: {str(e)}")


def calculate_roc_auc(coco_gt, pred_path, class_id=1):
    """Вычисляет ROC AUC для указанного класса"""
    # Загрузка предсказаний
    coco_pred = coco_gt.loadRes(pred_path)

    # Инициализация COCOeval
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.iouThrs = np.array([iou_threshold])  # Фиксируем IoU threshold

    # Оцениваем только указанный класс
    coco_eval.params.catIds = [class_id]

    # Вычисление метрик
    coco_eval.evaluate()
    coco_eval.accumulate()

    # print(coco_eval)

    # Сбор оценок и меток
    scores = []
    labels = []

    used_ids = []
    dublicates = 0

    for image_index, eval_img in enumerate(coco_eval.evalImgs):
        if eval_img is None:
            continue

        img_scores = eval_img['dtScores']
        img_matches = eval_img['dtMatches'][0]

        eval_lbs = []
        for id in img_matches:
            if id>0 and (image_index, id) not in used_ids:
                used_ids.append((image_index, id))
                eval_lbs.append(1)
            else:
                eval_lbs.append(0)
        labels.extend(eval_lbs)

        # labels.extend([1 if m > 0 else 0 for m in img_matches])
        scores.extend(img_scores)



    # Преобразование в numpy массивы
    scores = np.array(scores)
    labels = np.array(labels)
    print('#'*100 + f'{len(scores)=}')

    pos_scores = []
    for lb, sc in zip(labels, scores):
        if lb:
            pos_scores.append(sc)


    print(f"Средний confidence score при {iou_threshold=}: ", np.mean(pos_scores))

    import seaborn as sns
    sns.histplot(pos_scores, bins=25)
    plt.title(f'scores {iou_threshold=}')
    plt.show()


    pos_counter = 0
    neg_counter = 0
    for lb in labels:
        if lb:
            pos_counter += 1
        else:
            neg_counter += 1

    print(f'{pos_counter=}')
    print(f'{neg_counter=}')

    print(f'{len(labels)=}')

    # Вычисление ROC кривой
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)

    print(f'{len(fpr)=}')

    return fpr, tpr, roc_auc




def plot_roc_curve(fpr, tpr, roc_auc):
    """Визуализация ROC кривой"""
    title = f'{dataset_name} {model_name} IoU={iou_threshold}'

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title('ROC '+title)
    plt.legend(loc="lower right")
    plt.savefig(r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\ROC'+'\\'+title +'.png')
    plt.show()


# Основной скрипт
if __name__ == "__main__":
    # # Пути к файлам
    gt_json_path = gt_pathes[dataset_name]
    pred_json_path = pred_pathes[predict_name]
    pred_json_path = "FOMO_50e_predictions_UPDATE.json"


    # gt_json_path = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\vid1\vid1.json' #
    # pred_json_path = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\runs\detect\val y=12n p=1088 vid1\coco_predictions.json' # vid1 baseline

        # gt_json_path = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\val\mva23_val_UPDATED.json' #
    # pred_json_path = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\runs\detect\val y=12n p=640 SOD23_val\coco_predictions.json'  # Ваши предсказания

    # pred_json_path = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\submit\mva23_val\coco_predictions.json'  # Ваши предсказания

    try:
        # Загрузка и валидация данных
        coco_gt, temp_pred_path = load_and_validate_coco(gt_json_path, pred_json_path)

        # Вычисление ROC AUC (укажите нужный class_id)
        fpr, tpr, roc_auc = calculate_roc_auc(coco_gt, temp_pred_path, class_id=1)

        # Визуализация
        print(f"ROC AUC: {roc_auc:.4f}")
        plot_roc_curve(fpr, tpr, roc_auc)

    except Exception as e:
        raise e
        print(f"Ошибка: {str(e)}")
    finally:
        # Удаление временного файла
        import os

        if os.path.exists('temp_predictions.json'):
            os.remove('temp_predictions.json')