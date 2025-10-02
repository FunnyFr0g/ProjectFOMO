import os
import json
import argparse
from pathlib import Path
from clearml import Dataset, Task, StorageManager
import cv2


def create_coco_structure():
    """Создает базовую структуру COCO JSON"""
    return {
        "images": [],
        "annotations": [],
        "categories": []
    }


def yolo_to_coco_by_split(yolo_dataset_path, output_dir):
    """
    Преобразует YOLO разметку в COCO формат с отдельными JSON файлами для каждой выборки

    Args:
        yolo_dataset_path (str): Путь к датасету YOLO
        output_dir (str): Директория для сохранения JSON файлов
    """

    # Пути к данным YOLO
    images_dir = Path(yolo_dataset_path) / "images"
    labels_dir = Path(yolo_dataset_path) / "labels"

    # Определяем доступные выборки
    splits = []
    if (images_dir / "train").exists():
        splits = ["train", "val"]
    else:
        splits = [""]  # Если нет подпапок, используем корень

    # Собираем все классы из всего датасета
    all_classes = set()
    for split in splits:
        split_labels_dir = labels_dir / split if split else labels_dir
        if split_labels_dir.exists():
            for label_file in split_labels_dir.glob("*.txt"):
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.strip().split()[0])
                            all_classes.add(class_id)

    # Создаем категории (сдвигаем ID на 1)
    categories = []
    for class_id in sorted(list(all_classes)):
        categories.append({
            "id": class_id + 1,  # COCO начинается с 1
            "name": f"class_{class_id}",
            "supercategory": "none"
        })

    # Обрабатываем каждую выборку отдельно
    for split in splits:
        split_images_dir = images_dir / split if split else images_dir
        split_labels_dir = labels_dir / split if split else labels_dir

        if not split_images_dir.exists():
            print(f"Директория {split_images_dir} не существует, пропускаем")
            continue

        # Создаем структуру COCO для текущей выборки
        coco_data = create_coco_structure()
        coco_data["categories"] = categories

        annotation_id = 1  # Общий ID аннотаций для всей выборки
        image_id = 1  # ID изображений для текущей выборки

        # Получаем список изображений
        image_files = list(split_images_dir.glob("*.*"))
        image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

        print(f"Обрабатываем выборку '{split if split else 'all'}': {len(image_files)} изображений")

        for image_file in image_files:
            annotation_id = 1  # Общий ID аннотаций для всей выборки
            # Читаем размеры изображения
            img = cv2.imread(str(image_file))
            if img is None:
                print(f"Не удалось загрузить изображение: {image_file}")
                continue

            height, width = img.shape[:2]

            # Определяем путь к файлу для COCO
            if split:
                file_name_in_coco = f"{split}/{image_file.name}"
            else:
                file_name_in_coco = image_file.name

            # Добавляем информацию об изображении
            coco_data["images"].append({
                "id": image_id,
                "file_name": file_name_in_coco,
                "width": width,
                "height": height
            })

            # Соответствующий файл разметки
            label_file = split_labels_dir / f"{image_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()

                # Нумерация аннотаций для текущего изображения начинается с 1
                image_annotation_start_id = annotation_id

                for line_num, line in enumerate(lines, 1):
                    if line.strip():
                        data = line.strip().split()
                        if len(data) >= 5:
                            class_id = int(data[0])
                            x_center = float(data[1])
                            y_center = float(data[2])
                            bbox_width = float(data[3])
                            bbox_height = float(data[4])

                            # Преобразуем из YOLO формата в COCO
                            x_min = (x_center - bbox_width / 2) * width
                            y_min = (y_center - bbox_height / 2) * height
                            bbox_width_px = bbox_width * width
                            bbox_height_px = bbox_height * height

                            # Убеждаемся, что координаты не выходят за границы
                            x_min = max(0, x_min)
                            y_min = max(0, y_min)
                            bbox_width_px = min(width - x_min, bbox_width_px)
                            bbox_height_px = min(height - y_min, bbox_height_px)

                            # Добавляем аннотацию
                            coco_data["annotations"].append({
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": class_id + 1,  # COCO начинается с 1
                                "bbox": [x_min, y_min, bbox_width_px, bbox_height_px],
                                "area": bbox_width_px * bbox_height_px,
                                "iscrowd": 0,
                                "segmentation": []
                            })

                            annotation_id += 1

                print(
                    f"  Изображение {image_file.name}: {len(lines)} аннотаций (ID: {image_annotation_start_id}-{annotation_id - 1})")
            else:
                print(f"  Изображение {image_file.name}: нет файла разметки")

            image_id += 1

        # Сохраняем COCO JSON для текущей выборки
        output_json_path = Path(output_dir) / f"coco_annotations_{split if split else 'all'}.json"
        with open(output_json_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        print(f"COCO JSON для выборки '{split if split else 'all'}' сохранен в: {output_json_path}")
        print(f"  Изображений: {len(coco_data['images'])}")
        print(f"  Аннотаций: {len(coco_data['annotations'])}")
        print(f"  Категорий: {len(coco_data['categories'])}")

    return output_dir


def main_clearml():
    """Основная функция для запуска в ClearML"""

    Task.ignore_requirements('pywin32')
    task = Task.init(
        project_name="SmallObjectDetection",
        task_name="Convert YOLO dataset to COCO format with split JSONs",
        task_type=Task.TaskTypes.data_processing
    )

    task.execute_remotely(queue_name='default', exit_process=True)


    args = {'dataset_id' : 'ae8c12c33b324947af9ae6379d920eb8',
            'output_dir' : 'tmp'
            }

    # Получаем датасет
    print(f"Получаем датасет: {args['dataset_id']}")
    dataset = Dataset.get(dataset_id=args['dataset_id'])
    dataset_path = dataset.get_local_copy()

    print(f"Датасет загружен в: {args['dataset_id']}")

    # Создаем директорию для выходных файлов
    os.makedirs(args['dataset_id'], exist_ok=True)

    # Преобразуем YOLO в COCO
    output_dir = yolo_to_coco_by_split(dataset_path, args['dataset_id'])

    # Сохраняем все JSON файлы как артефакты
    for json_file in Path(output_dir).glob("*.json"):
        task.upload_artifact(
            name=json_file.stem,
            artifact_object=str(json_file),
            metadata={
                "format": "COCO",
                "source": "YOLO",
                "file_size": os.path.getsize(json_file)
            }
        )
        print(f"Артефакт загружен: {json_file.name}")

    print("Задача завершена успешно!")


def main_local():
    """Функция для локального запуска из IDE"""
    dataset_id = "YOUR_DATASET_ID_HERE"  # Замените на ваш dataset ID
    output_dir = "./coco_output"

    # Получаем датасет
    print(f"Получаем датасет: {dataset_id}")
    dataset = Dataset.get(dataset_id=dataset_id)
    dataset_path = dataset.get_local_copy()

    print(f"Датасет загружен в: {dataset_path}")

    # Создаем директорию для выходных файлов
    os.makedirs(output_dir, exist_ok=True)

    # Преобразуем YOLO в COCO
    yolo_to_coco_by_split(dataset_path, output_dir)

    print(f"Все JSON файлы сохранены в директории: {output_dir}")


if __name__ == "__main__":
    try:
        from clearml import Task

        print("Запуск в режиме ClearML...")
        main_clearml()
    except ImportError:
        print("ClearML не найден, запуск в локальном режиме...")
        main_local()