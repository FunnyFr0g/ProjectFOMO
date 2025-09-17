import os
import json
import argparse
from PIL import Image
from collections import defaultdict


def convert_yolo_to_coco(yolo_dir, output_file, class_file=None):
    """
    Конвертирует набор данных с разметкой YOLO в формат COCO.

    Параметры:
        yolo_dir (str): Путь к директории с данными YOLO (images/, labels/)
        output_file (str): Путь для сохранения JSON файла COCO
        class_file (str, optional): Путь к файлу с классами (если нет classes.txt)
    """
    # Пути к директориям
    images_dir = os.path.join(yolo_dir, 'images')
    labels_dir = os.path.join(yolo_dir, 'labels')

    # Проверка существования директорий
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise ValueError("Директории 'images' и 'labels' должны находиться в yolo_dir")

    # Загрузка классов
    if isinstance(class_file, (tuple, list)):
        classes = class_file

    elif class_file is None:
        class_file = os.path.join(yolo_dir, 'classes.txt')

    elif not os.path.exists(class_file):
        print(f"Файл с классами не найден: {class_file}")
        classes = [f'cl_{i}' for i in range(0, 10)]
    else:
        with open(class_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

    # Инициализация структуры COCO
    coco = {
        "info": {
            "description": "Dataset converted from YOLO format",
            "version": "1.0",
            "year": 2023,
            "contributor": "",
            "date_created": "2023-01-01"
        },
        "licenses": [{
            "id": 1,
            "name": "Unknown",
            "url": ""
        }],
        "categories": [],
        "images": [],
        "annotations": []
    }

    # Добавление категорий
    for i, class_name in enumerate(classes, 1):
        coco["categories"].append({
            "id": i,
            "name": class_name,
            "supercategory": "none"
        })

    # Сопоставление файлов изображений и аннотаций
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    annotation_files = [f.replace(os.path.splitext(f)[1], '.txt') for f in image_files]

    # Счетчики
    image_id = 1
    annotation_id = 1

    # Обработка каждого изображения
    for img_file, ann_file in zip(image_files, annotation_files):
        # Полные пути к файлам
        img_path = os.path.join(images_dir, img_file)
        ann_path = os.path.join(labels_dir, ann_file)

        # Пропускаем если нет аннотации
        if not os.path.exists(ann_path):
            continue

        # Получение размеров изображения
        with Image.open(img_path) as img:
            width, height = img.size

        # Добавление информации об изображении
        coco["images"].append({
            "id": image_id,
            "file_name": img_file,
            "width": width,
            "height": height,
            "license": 1,
            "date_captured": "2023-01-01"
        })

        # Чтение аннотаций YOLO
        with open(ann_path, 'r') as f:
            lines = f.readlines()

        # Обработка каждой аннотации
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            # Конвертация из YOLO в COCO формат
            x_min = (x_center - box_width / 2) * width
            y_min = (y_center - box_height / 2) * height
            box_width = box_width * width
            box_height = box_height * height

            # Добавление аннотации
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id + 1,  # COCO использует 1-based индексацию
                "bbox": [x_min, y_min, box_width, box_height],
                "area": box_width * box_height,
                "segmentation": [],
                "iscrowd": 0
            })

            annotation_id += 1

        image_id += 1

    # Сохранение JSON файла
    with open(output_file, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"Конвертация завершена. Результат сохранен в {output_file}")
    print(f"Всего изображений: {len(coco['images'])}")
    print(f"Всего аннотаций: {len(coco['annotations'])}")
    print(f"Количество классов: {len(classes)}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Конвертация разметки YOLO в COCO формат')
    # parser.add_argument('--yolo_dir', type=str, required=True,
    #                     help='Путь к директории с данными YOLO (содержащей images/ и labels/)')
    # parser.add_argument('--output', type=str, required=True, help='Путь для сохранения JSON файла COCO')
    # parser.add_argument('--classes', type=str, help='Путь к файлу с классами (если не classes.txt)')
    #
    # args = parser.parse_args()
    #
    # convert_yolo_to_coco(args.yolo_dir, args.output, args.classes)

    yolo_dir = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test'
    coco_dir = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test\skb_test.json'

    convert_yolo_to_coco(yolo_dir, coco_dir, ('bird',))
