import os
import json
import cv2
import numpy as np
from torchvision import transforms
import torch
from torchvision.models import mobilenet_v2
import torch.nn as nn
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from skimage.measure import regionprops
from typing import List, Tuple, Dict, Any

# Константы
BOX_SIZE = 8  # Размер стороны квадратного bounding box'а в пикселях
TRUNK_AT = 4
NUM_CLASSES = 2
TILE_SIZE = 800  # Размер тайла (должен соответствовать входному размеру модели)
OVERLAP_RATIO = 0.1  # Перекрытие между тайлами


class FomoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = mobilenet_v2(pretrained=False).features[:TRUNK_AT]

    def forward(self, x):
        return self.mobilenet(x)


class FomoHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(24, 48, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(48, 32, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        return x


class FomoModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = FomoBackbone()
        self.head = FomoHead(num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def get_slice_bboxes(
        image_height: int,
        image_width: int,
        slice_height: int = TILE_SIZE,
        slice_width: int = TILE_SIZE,
        overlap_height_ratio: float = OVERLAP_RATIO,
        overlap_width_ratio: float = OVERLAP_RATIO,
) -> List[List[int]]:
    """Разделяет изображение на перекрывающиеся тайлы.

    Args:
        image_height: Высота исходного изображения
        image_width: Ширина исходного изображения
        slice_height: Высота тайла
        slice_width: Ширина тайла
        overlap_height_ratio: Доля перекрытия по высоте (0-1)
        overlap_width_ratio: Доля перекрытия по ширине (0-1)

    Returns:
        Список координат тайлов в формате [xmin, ymin, xmax, ymax]
    """
    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)

    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


def prepare_tile(image_tile: np.ndarray) -> torch.Tensor:
    """Подготовка тайла изображения для модели"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image_tile)
    return image_tensor.unsqueeze(0)


def process_predictions(
        pred_mask: np.ndarray,
        pred_probs: np.ndarray,
        orig_size: Tuple[int, int],
        image_id: int,
        offset_x: int = 0,
        offset_y: int = 0
) -> List[Dict[str, Any]]:
    """Обработка предсказаний с добавлением confidence score и объединением близких пикселей"""
    annotations = []
    annotation_id = 1

    for class_id in range(1, NUM_CLASSES):
        class_mask = (pred_mask == class_id).astype(np.uint8)

        if len(class_mask.shape) > 2:
            class_mask = class_mask.squeeze()

        cleaned_mask = class_mask

        # Находим все ненулевые пиксели для текущего класса
        y_coords, x_coords = np.where(cleaned_mask == 1)

        if len(y_coords) == 0:
            continue

        # Создаем список точек и соответствующих scores
        points = list(zip(y_coords, x_coords))
        scores = pred_probs[class_id][y_coords, x_coords]

        # Группируем близлежащие пиксели (в радиусе 4x4)
        processed = np.zeros(len(points), dtype=bool)
        groups = []

        for i in range(len(points)):
            if not processed[i]:
                current_group = [i]
                processed[i] = True
                queue = [i]

                while queue:
                    idx = queue.pop(0)
                    y1, x1 = points[idx]

                    for j in range(len(points)):
                        if not processed[j]:
                            y2, x2 = points[j]
                            if abs(y1 - y2) <= 2 and abs(x1 - x2) <= 2:
                                processed[j] = True
                                current_group.append(j)
                                queue.append(j)

                groups.append(current_group)

        # Обрабатываем каждую группу
        for group in groups:
            if not group:
                continue

            group_points = [points[i] for i in group]
            group_scores = [scores[i] for i in group]

            y_coords = [p[0] for p in group_points]
            x_coords = [p[1] for p in group_points]

            y_centroid = np.mean(y_coords)
            x_centroid = np.mean(x_coords)
            score = np.mean(group_scores)

            # Масштабируем координаты
            scaled_coords = scale_coords([(y_centroid, x_centroid)],
                                         from_size=pred_mask.shape,
                                         to_size=(TILE_SIZE, TILE_SIZE),
                                         orig_size=orig_size)
            y_orig, x_orig = scaled_coords[0]

            # Добавляем смещение тайла
            y_orig += offset_y
            x_orig += offset_x

            # Bounding box
            half_size = BOX_SIZE // 2
            x1 = max(0, x_orig - half_size)
            y1 = max(0, y_orig - half_size)
            x2 = min(orig_size[0], x_orig + half_size)
            y2 = min(orig_size[1], y_orig + half_size)

            area = len(group) * (orig_size[0] / pred_mask.shape[1]) * (orig_size[1] / pred_mask.shape[0])

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": area,
                "score": float(score),
                "iscrowd": 0,
                "centroid": [x_orig, y_orig]
            }

            annotations.append(annotation)
            annotation_id += 1

    return annotations


def scale_coords(coords, from_size=(56, 56), to_size=(224, 224), orig_size=None):
    """Масштабирование координат из feature map в оригинальное разрешение"""
    if orig_size is None:
        orig_size = to_size

    x_scale = to_size[0] / from_size[0]
    y_scale = to_size[1] / from_size[1]

    x_scale *= orig_size[0] / to_size[0]
    y_scale *= orig_size[1] / to_size[1]

    scaled_coords = []
    for y, x in coords:
        scaled_coords.append((int(y * y_scale), int(x * x_scale)))
    return scaled_coords


def save_to_coco_format(images, annotations, output_path):
    """Сохранение результатов в COCO-формате"""
    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "bird"}],
        "info": {"description": "FOMO model predictions"},
        "licenses": [{"id": 1, "name": "CC-BY"}]
    }

    with open(output_path, 'w') as f:
        json.dump(coco_output, f, indent=2)


def process_image_with_tiling(model: nn.Module, img_path: str, image_id: int) -> Tuple[Dict, List[Dict]]:
    """Обработка изображения с использованием тайлинга"""
    # Загрузка изображения
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]
    orig_size = (orig_w, orig_h)

    # Получение координат тайлов
    slice_bboxes = get_slice_bboxes(orig_h, orig_w)

    # Создаем запись об изображении
    image_info = {
        "id": image_id,
        "file_name": os.path.basename(img_path),
        "width": orig_w,
        "height": orig_h
    }

    all_annotations = []

    # Обработка каждого тайла
    for bbox in slice_bboxes:
        xmin, ymin, xmax, ymax = bbox
        tile = image[ymin:ymax, xmin:xmax]

        # Подготовка и предсказание
        tile_tensor = prepare_tile(tile)
        with torch.no_grad():
            output = model(tile_tensor)
            probs = F.softmax(output, dim=1).squeeze(0).cpu().numpy()
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # Обработка предсказаний с учетом смещения тайла
        tile_annotations = process_predictions(
            pred_mask, probs, (xmax - xmin, ymax - ymin),
            image_id, offset_x=xmin, offset_y=ymin
        )
        all_annotations.extend(tile_annotations)

    return image_info, all_annotations


# Основной код
model = FomoModel(num_classes=NUM_CLASSES)
checkpoint_path = 'FOMO_56_focalloss_50e_model_weights.pth'
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

img_dir = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test\images'
output_json = os.path.join(img_dir, f'FOMO_50e_SAHI_{TILE_SIZE}p_predictions_tiled.json')

all_annotations = []
all_images = []
image_id = 0

import time

start_time = time.time()

for img_name in os.listdir(img_dir):
    image_id += 1
    # image_id = int(img_name.strip('.jpg'))
    img_path = os.path.join(img_dir, img_name)

    # Обработка изображения с тайлингом
    image_info, annotations = process_image_with_tiling(model, img_path, image_id)

    all_images.append(image_info)
    all_annotations.extend(annotations)
    print(f"Processed image {image_id}: {img_name}")

end_time = time.time()
elapsed_time = end_time - start_time

print(elapsed_time)
print(elapsed_time/image_id)
# Сохраняем результаты
# save_to_coco_format(all_images, all_annotations, output_json)
print(f"Predictions saved to {output_json}")