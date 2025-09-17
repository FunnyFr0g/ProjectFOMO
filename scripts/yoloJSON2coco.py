import json
from pathlib import Path

import json
from pathlib import Path


def yolo_predictions_to_coco(
        yolo_json_path: str,
        existing_coco_path: str,
        output_coco_path: str,
        class_names: list
):
    """
    Конвертирует predictions.json от YOLO в COCO-формат, сохраняя оригинальные image_id из существующего COCO-файла.

    :param yolo_json_path: Путь к predictions.json от YOLO (с именами файлов в image_id)
    :param existing_coco_path: Путь к вашему исходному COCO-файлу с разметкой
    :param output_coco_path: Путь для сохранения результата
    :param class_names: Список названий классов (например, ['cat', 'dog'])
    """
    # Загружаем существующий COCO-файл для маппинга file_name → image_id
    with open(existing_coco_path) as f:
        existing_coco = json.load(f)

    # Создаем словарь file_name → image_id из существующего COCO
    file_to_id = {img["file_name"]: img["id"] for img in existing_coco["images"]}

    # Загружаем предсказания YOLO
    with open(yolo_json_path) as f:
        yolo_preds = json.load(f)

    # Создаем структуру COCO для результата
    coco = {
        "info": existing_coco.get("info", {}),
        "licenses": existing_coco.get("licenses", []),
        "categories": [
            {"id": i + 1, "name": name, "supercategory": "object"}
            for i, name in enumerate(class_names)
        ],
        "images": existing_coco["images"],  # Сохраняем оригинальные изображения
        "annotations": []
    }

    # Обрабатываем каждое предсказание YOLO
    for ann_id, pred in enumerate(yolo_preds, 1):
        file_name = str(pred["image_id"])+ '.jpg' #'.png' # '.jpg' # YOLO 00011.jpg превращает в 1 => нужно обратно

        # Пропускаем, если файла нет в исходном COCO
        if file_name not in file_to_id:
            print(f"Warning: {file_name} не найден в existing_coco.json, пропускаем")
            continue

        coco["annotations"].append({
            "id": ann_id,
            "image_id": file_to_id[file_name],  # Используем оригинальный image_id
            "category_id": pred["category_id"],  # COCO использует 1-based ВНИМАТЕЛЬНО
            "bbox": pred["bbox"],
            "area": pred["bbox"][2] * pred["bbox"][3],
            "iscrowd": 0,
            "score": pred.get("score", 0.0)  # Опционально (не в стандарте COCO)
        })

    # Сохраняем результат
    with open(output_coco_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"Готово! Результат сохранен в {output_coco_path}")
    print(f"Всего аннотаций: {len(coco['annotations'])}")



# yolo_predictions_to_coco(
#     yolo_json_path=r"X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\runs\detect\val y=12n p=640 SOD23_val\predictions.json",
#     existing_coco_path=r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\val\mva23_val_UPDATED.json',
#     output_coco_path=r"X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\runs\detect\val y=12n p=640 SOD23_val\coco_predictions.json",
#     class_names=['bird'],  # Ваши классы,
# )

# yolo_predictions_to_coco(
# yolo_json_path=r"X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\runs\detect\val y=12n p=1088 vid1\predictions.json",
#     existing_coco_path=r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\vid1\vid1.json',
#     output_coco_path=r"X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\runs\detect\val y=12n p=1088 vid1\coco_predictions.json",
#     class_names=['bird'],  # Ваши классы,
# )

from label_pathes import *

yolo_predictions_to_coco(
yolo_json_path=r"FOMO_50e_predictions.json",
    existing_coco_path=gt_pathes['mva23_val_FOMO'],
    output_coco_path='FOMO_50e_predictions_UPDATE.json',
    class_names=['bird'],  # Ваши классы,
)