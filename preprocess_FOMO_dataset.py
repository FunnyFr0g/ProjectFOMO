import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import shutil
from pycocotools.coco import COCO
from typing import Dict, List, Tuple
from clearml import Task, Dataset

task = Task.init(project_name='SmallObjectDetection', task_name='drones_only Preprocessing FOMO', tags=['FOMO'])


task.execute_remotely(queue_name='default', exit_process=True)

params = {
    "min_bbox" : 4,
    "max_bbox" : 16,
}

params = task.connect(params)

random.seed(42)


class COCOBboxResizer:
    def __init__(
        self,
        coco_annotation_path: str,
        image_dir: str,
        output_dir: str,
        min_bbox: int,
        max_bbox: int,
        target_image_size: Tuple[int, int] = (640, 640),
        jitter: float = 0.2,
        need_masks=True,
        json_name="annotations.json",
    ):
        """
        Args:
            coco_annotation_path: Path to COCO annotation file
            image_dir: Directory with images
            output_dir: Directory to save processed images and annotations
            min_bbox: Minimum bbox size (in pixels) after processing
            max_bbox: Maximum bbox size (in pixels) after processing
            target_image_size: Size of output images (width, height)
            jitter: Relative jitter for crop position (0-1)
        """
        self.coco_annotation_path = coco_annotation_path
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.min_bbox = min_bbox
        self.max_bbox = max_bbox
        self.target_width, self.target_height = target_image_size
        self.jitter = jitter
        self.need_masks = need_masks
        self.json_name = json_name

        # Create output directories
        self.output_image_dir = os.path.join(output_dir, "images")
        os.makedirs(self.output_image_dir, exist_ok=True)
        if need_masks:
            self.output_mask_dir = os.path.join(output_dir, "fomo_masks")
            os.makedirs(self.output_mask_dir, exist_ok=True)

        # Load original COCO dataset
        self.coco = COCO(coco_annotation_path)
        self.original_images = self.coco.dataset["images"]
        self.original_annotations = self.coco.dataset["annotations"]
        self.original_categories = self.coco.dataset["categories"]

        # Prepare new COCO dataset
        self.new_dataset = {
            "images": [],
            "annotations": [],
            "categories": self.original_categories,
        }
        self.annotation_id = 1

    def _calculate_scale_factor(self, bbox_width: float, bbox_height: float) -> float:
        """Calculate scale factor to resize bbox to target size range."""
        bbox_size = max(bbox_width, bbox_height)
        target_size = random.uniform(self.min_bbox, self.max_bbox)
        return target_size / bbox_size

    def _get_random_crop_position(
        self, bbox: List[float], scaled_width: int, scaled_height: int
    ) -> Tuple[int, int]:
        """Calculate random crop position with jitter around the bbox center."""
        bbox_center_x = bbox[0] + bbox[2] / 2
        bbox_center_y = bbox[1] + bbox[3] / 2

        # Convert bbox center to scaled image coordinates
        scaled_center_x = bbox_center_x * self.scale_factor
        scaled_center_y = bbox_center_y * self.scale_factor

        # Calculate max jitter distance
        max_jitter_x = self.jitter * self.target_width
        max_jitter_y = self.jitter * self.target_height

        # Calculate crop position with jitter
        crop_x = max(
            0,
            min(
                scaled_center_x - self.target_width / 2 + random.uniform(-max_jitter_x, max_jitter_x),
                scaled_width - self.target_width,
            ),
        )
        crop_y = max(
            0,
            min(
                scaled_center_y - self.target_height / 2 + random.uniform(-max_jitter_y, max_jitter_y),
                scaled_height - self.target_height,
            ),
        )

        return int(crop_x), int(crop_y)

    def _create_fomo_mask(
        self, annotations: List[Dict], crop_x: int, crop_y: int, image_id: int
    ) -> None:
        """Create FOMO mask (8-bit grayscale where pixel value = class_id)."""
        mask = Image.new("L", (self.target_width, self.target_height), 0)
        draw = ImageDraw.Draw(mask)

        for ann in annotations:
            if ann["image_id"] != image_id:
                continue

            # Scale and shift bbox
            x1 = ann["bbox"][0] * self.scale_factor - crop_x
            y1 = ann["bbox"][1] * self.scale_factor - crop_y
            x2 = x1 + ann["bbox"][2] * self.scale_factor
            y2 = y1 + ann["bbox"][3] * self.scale_factor

            # Clip to image bounds
            x1 = max(0, min(x1, self.target_width))
            y1 = max(0, min(y1, self.target_height))
            x2 = max(0, min(x2, self.target_width))
            y2 = max(0, min(y2, self.target_height))

            if x1 >= x2 or y1 >= y2:
                continue

            # Draw rectangle with class_id as pixel value
            draw.rectangle([x1, y1, x2, y2], fill=ann["category_id"]+1)

        mask.save(os.path.join(self.output_mask_dir, f"{image_id}.png"))

    def process_image(self, image_info: Dict) -> None:
        """Process single image."""
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        image_id = image_info["id"]

        # Load image
        img = Image.open(image_path).convert("RGB")
        original_width, original_height = img.size

        # Get all annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        if not annotations:
            return

        # Select a random annotation to determine scale
        target_ann = random.choice(annotations)
        bbox = target_ann["bbox"]

        # Calculate scale factor based on the largest dimension of the bbox
        self.scale_factor = self._calculate_scale_factor(bbox[2], bbox[3])

        # Scale image
        scaled_width = int(original_width * self.scale_factor)
        scaled_height = int(original_height * self.scale_factor)
        scaled_img = img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)

        # Calculate random crop position
        crop_x, crop_y = self._get_random_crop_position(bbox, scaled_width, scaled_height)

        # Crop image
        cropped_img = scaled_img.crop(
            (
                crop_x,
                crop_y,
                crop_x + self.target_width,
                crop_y + self.target_height,
            )
        )

        # Save processed image
        # new_image_name = f"{image_id}.jpg"
        new_image_name = image_info["file_name"]
        cropped_img.save(os.path.join(self.output_image_dir, new_image_name))

        # Create FOMO mask
        if self.need_masks:
            self._create_fomo_mask(annotations, crop_x, crop_y, image_id)

        # Prepare new image info for COCO dataset
        new_image_info = {
            "id": image_id,
            "file_name": new_image_name,
            "width": self.target_width,
            "height": self.target_height,
        }
        self.new_dataset["images"].append(new_image_info)

        # Process and add annotations
        for ann in annotations:
            # Scale and shift bbox
            x1 = ann["bbox"][0] * self.scale_factor - crop_x
            y1 = ann["bbox"][1] * self.scale_factor - crop_y
            w = ann["bbox"][2] * self.scale_factor
            h = ann["bbox"][3] * self.scale_factor

            # Clip to image bounds
            x1 = max(0, min(x1, self.target_width))
            y1 = max(0, min(y1, self.target_height))
            w = max(0, min(w, self.target_width - x1))
            h = max(0, min(h, self.target_height - y1))

            if w <= 0 or h <= 0:
                continue

            new_ann = {
                "id": self.annotation_id,
                "image_id": image_id,
                "category_id": ann["category_id"]+1, ######### ВАЖНО чтоб ID шел с 1 как в coco
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": ann["iscrowd"],
            }
            self.new_dataset["annotations"].append(new_ann)
            self.annotation_id += 1

    def process_dataset(self) -> None:
        """Process all images in the dataset."""
        for image_info in tqdm(self.original_images, desc="Processing images"):
            try:
                self.process_image(image_info)
            except Exception as e:
                print(f"Error processing image {image_info['id']}: {e}")

        # Save new COCO annotations
        output_annotation_path = os.path.join(self.output_dir, self.json_name)
        with open(output_annotation_path, "w") as f:
            json.dump(self.new_dataset, f)

        print(f"Processing complete. Results saved to {self.output_dir}")



def convert_yolo_to_coco(images_dir, labels_dir, output_file_path, class_file=None, force=False):
    """
    Конвертирует набор данных с разметкой YOLO в формат COCO.

    Параметры:
        images_dir (str): Путь к директории с фото
        labels_dir (str): Путь к директории с метками
        output_file_path (str): Путь для сохранения JSON файла COCO
        class_file (str, optional): Путь к файлу с классами (если нет classes.txt)
        force (bool, optional): пересоздать файл COCO с разметкой, если он существует
    """
    # Пути к директориям
    # images_dir = os.path.join(yolo_dir, 'images')
    # labels_dir = os.path.join(yolo_dir, 'labels')

    # Проверка существования директорий
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise ValueError("Директории 'images' и 'labels' должны находиться в yolo_dir")

    if os.path.exists(output_file_path) and not force:
        print(f"Output file {output_file_path} already exists, skipping.")
        return output_file_path



    # Загрузка классов
    if isinstance(class_file, (tuple, list)):
        classes = class_file

    elif class_file is None:
        class_file = os.path.join(labels_dir, 'classes.txt')

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
            "year": 2025,
            "contributor": "",
            "date_created": "2025-01-01"
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
    with open(output_file_path, 'w') as f:
        json.dump(coco, f, indent=2)

    return output_file_path

    print(f"Конвертация завершена. Результат сохранен в {output_file_path}")
    print(f"Всего изображений: {len(coco['images'])}")
    print(f"Всего аннотаций: {len(coco['annotations'])}")
    print(f"Количество классов: {len(classes)}")


# Example usage
if __name__ == "__main__":
    #
    old_dataset_dir = Dataset.get(dataset_name="FOMO-mva23").get_local_copy()
    os.makedirs("processed_dataset/train", exist_ok=True)
    
    # preprocessing для train выборки
    train_image_dir = os.path.join(old_dataset_dir, "images", 'train')
    train_labels_dir = os.path.join(old_dataset_dir, "labels", 'train')
    train_json = os.path.join(old_dataset_dir, "train_annotations.json")
    train_coco_annotations = convert_yolo_to_coco(train_image_dir, train_labels_dir, train_json, ('drone',))
    

    # Configuration
    output_dir = r"processed_dataset/train"
    min_bbox = params['min_bbox']  # Minimum bbox size in pixels
    max_bbox = params['max_bbox']  # Maximum bbox size in pixels
    target_image_size = (224, 224)  # Output image size
    jitter = 0.2  # Relative jitter for crop position

    # Create and run processor
    train_processor = COCOBboxResizer(
        coco_annotation_path=train_coco_annotations,
        image_dir=train_image_dir,
        output_dir=output_dir,
        min_bbox=min_bbox,
        max_bbox=max_bbox,
        target_image_size=target_image_size,
        jitter=jitter,
        need_masks=False,
        json_name="train_annotations.json"
    )
    train_processor.process_dataset()
    
    
    
    os.makedirs("processed_dataset/val", exist_ok=True)
    
    # preprocessing для val выборки
    val_image_dir = os.path.join(old_dataset_dir, "images", 'val')
    val_labels_dir = os.path.join(old_dataset_dir, "labels", 'val')
    val_json = os.path.join(old_dataset_dir, "val_annotations.json")
    val_coco_annotations = convert_yolo_to_coco(val_image_dir, val_labels_dir, val_json, ('drone',))
    

    # Configuration
    output_dir = r"processed_dataset/val"
    min_bbox = 4  # Minimum bbox size in pixels
    max_bbox = 16  # Maximum bbox size in pixels
    target_image_size = (224, 224)  # Output image size
    jitter = 0.2  # Relative jitter for crop position

    # Create and run processor
    val_processor = COCOBboxResizer(
        coco_annotation_path=val_coco_annotations,
        image_dir=val_image_dir,
        output_dir=output_dir,
        min_bbox=min_bbox,
        max_bbox=max_bbox,
        target_image_size=target_image_size,
        jitter=jitter,
        need_masks=False,
        json_name="val_annotations.json"
    )
    val_processor.process_dataset()

    new_dataset = Dataset.create(dataset_name="drones_only_FOMO", dataset_project="SmallObjectDetection", dataset_version='1.0.0')
    new_dataset.add_files("processed_dataset")

    new_dataset.upload(compression=False)
    new_dataset.finalize()




