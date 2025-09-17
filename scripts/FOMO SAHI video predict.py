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
from sahi.slicing import slice_image
from sahi.postprocess.combine import (
    GreedyNMMPostprocess,
    NMSPostprocess,
    LSNMSPostprocess,
    NMMPostprocess,
)
from sahi.prediction import PredictionResult
from typing import List, Dict, Any

# Константы
BOX_SIZE = 8  # Размер стороны квадратного bounding box'а в пикселях
TRUNK_AT = 4
NUM_CLASSES = 2
TILE_SIZE = 448  # Размер тайлов для SAHI
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


def prepare_frame(frame):
    """Подготовка кадра для модели (без изменения размера)"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    frame_tensor = transform(frame)
    return frame_tensor.unsqueeze(0)


def scale_coords(coords, from_size, to_size):
    """Масштабирование координат из одного разрешения в другое"""
    x_scale = to_size[0] / from_size[0]
    y_scale = to_size[1] / from_size[1]

    scaled_coords = []
    for y, x in coords:
        scaled_coords.append((int(y * y_scale), int(x * x_scale)))
    return scaled_coords


def process_tile_predictions(pred_mask, pred_probs, tile_bbox, orig_size, image_id):
    """Обработка предсказаний для одного тайла"""
    annotations = []
    annotation_id = 1  # Будем увеличивать для каждого тайла

    for class_id in range(1, NUM_CLASSES):
        class_mask = (pred_mask == class_id).astype(np.uint8)

        if len(class_mask.shape) > 2:
            class_mask = class_mask.squeeze()

        # Находим все ненулевые пиксели для текущего класса
        y_coords, x_coords = np.where(class_mask == 1)

        if len(y_coords) == 0:
            continue

        # Создаем список точек и соответствующих scores
        points = list(zip(y_coords, x_coords))
        scores = pred_probs[class_id][y_coords, x_coords]

        # Группируем близлежащие пиксели (в радиусе 2x2)
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

            # Масштабируем координаты к размеру тайла
            y_centroid = np.mean(y_coords)
            x_centroid = np.mean(x_coords)
            score = np.mean(group_scores)

            # Масштабируем координаты к оригинальному разрешению тайла
            tile_h, tile_w = pred_mask.shape
            scaled_tile_coords = scale_coords(
                [(y_centroid, x_centroid)],
                from_size=(tile_h, tile_w),
                to_size=(TILE_SIZE, TILE_SIZE)
            )
            y_tile, x_tile = scaled_tile_coords[0]

            # Переводим координаты в систему координат исходного изображения
            x_orig = tile_bbox[0] + x_tile
            y_orig = tile_bbox[1] + y_tile

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


def process_frame_with_sahi(frame, model, image_id):
    """Обработка кадра с использованием SAHI: разбиение на тайлы и объединение результатов"""
    orig_h, orig_w = frame.shape[:2]
    orig_size = (orig_w, orig_h)

    # Разбиваем изображение на тайлы с перекрытием
    slice_result = slice_image(
        image=frame,
        slice_height=TILE_SIZE,
        slice_width=TILE_SIZE,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO,
    )

    all_annotations = []

    for tile_dict in slice_result:
        # Получаем изображение тайла и его координаты
        tile_image = tile_dict["image"]
        # В новых версиях SAHI координаты хранятся в "starting_pixel" (y, x)
        tile_start_y, tile_start_x = tile_dict["starting_pixel"]

        # Подготавливаем тайл для модели
        tile_tensor = prepare_frame(tile_image)

        with torch.no_grad():
            output = model(tile_tensor)
            probs = F.softmax(output, dim=1).squeeze(0).cpu().numpy()
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # Обрабатываем предсказания для тайла
        tile_annotations = process_tile_predictions(
            pred_mask=pred_mask,
            pred_probs=probs,
            tile_bbox=(tile_start_x, tile_start_y),  # (x, y)
            orig_size=orig_size,
            image_id=image_id
        )

        all_annotations.extend(tile_annotations)

    # Применяем NMS для удаления дубликатов на границах тайлов
    filtered_annotations = apply_nms(all_annotations)

    return filtered_annotations


def apply_nms(annotations: List[Dict[str, Any]], iou_threshold: float = 0.3) -> List[Dict[str, Any]]:
    """Применение Non-Maximum Suppression для удаления дублирующихся детекций"""
    if not annotations:
        return []

    # Преобразуем аннотации в формат, понятный для NMS
    boxes = []
    scores = []
    classes = []
    keep_indices = []

    for ann in annotations:
        x1, y1, w, h = ann["bbox"]
        boxes.append([x1, y1, x1 + w, y1 + h])
        scores.append(ann["score"])
        classes.append(ann["category_id"])

    if len(boxes) == 0:
        return []

    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)

    # Применяем NMS для каждого класса отдельно
    unique_classes = set(classes)
    for cls in unique_classes:
        cls_indices = [i for i, c in enumerate(classes) if c == cls]
        if not cls_indices:
            continue

        cls_boxes = boxes[cls_indices]
        cls_scores = scores[cls_indices]

        # Применяем NMS
        keep = torch.ops.torchvision.nms(cls_boxes, cls_scores, iou_threshold)
        keep_indices.extend([cls_indices[i] for i in keep])

    # Удаляем дубликаты
    keep_indices = list(set(keep_indices))
    filtered_annotations = [annotations[i] for i in keep_indices]

    # Обновляем ID аннотаций
    for i, ann in enumerate(filtered_annotations):
        ann["id"] = i + 1

    return filtered_annotations


def save_to_coco_format(images, annotations, output_path):
    """Сохранение результатов в COCO-формате"""
    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "bird"}],
        "info": {"description": "FOMO model predictions with SAHI"},
        "licenses": [{"id": 1, "name": "CC-BY"}]
    }

    with open(output_path, 'w') as f:
        json.dump(coco_output, f, indent=2)


def process_video(video_path, output_video, output_json, model, frame_interval=1):
    """Обработка видеофайла с сохранением результатов детекции"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    all_annotations = []
    all_images = []
    frame_count = 0
    image_id = 0

    # Получаем параметры видео для сохранения результата
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Инициализируем VideoWriter для сохранения результата
    if output_video.endswith('mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        image_id += 1
        orig_h, orig_w = frame.shape[:2]

        # Создаем запись о кадре
        image_info = {
            "id": image_id,
            "file_name": f"frame_{frame_count:06d}.jpg",
            "width": orig_w,
            "height": orig_h,
            "frame_number": frame_count
        }
        all_images.append(image_info)

        # Обрабатываем кадр с SAHI
        annotations = process_frame_with_sahi(frame, model, image_id)
        all_annotations.extend(annotations)

        # Визуализация
        vis_frame = visualize_detections(frame, annotations, frame_count)
        out_video.write(vis_frame)

        # Показываем прогресс
        cv2.imshow('Detections', vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

    # Сохраняем результаты
    save_to_coco_format(all_images, all_annotations, output_json)
    print(f"Predictions saved to {output_json}")


def visualize_detections(frame, annotations, frame_count):
    """Визуализация детекций на кадре"""
    vis_frame = frame.copy()

    for ann in annotations:
        x1, y1, w, h = ann["bbox"]
        x2, y2 = x1 + w, y1 + h
        score = ann["score"]
        class_id = ann["category_id"]

        # Рисуем bounding box
        color = (0, 255, 0) if class_id == 1 else (0, 0, 255)
        cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Добавляем текст с score
        label = f"{class_id}: {score:.2f}"
        cv2.putText(vis_frame, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Добавляем номер кадра
    cv2.putText(vis_frame, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return vis_frame


def main():
    model = FomoModel(num_classes=NUM_CLASSES)
    checkpoint_path = 'FOMO_56_focalloss_50e_model_weights.pth'
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Обработка видеофайла
    vid_dir = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\vid1'
    vid_name = 'vid1.avi'
    video_path = os.path.join(vid_dir, vid_name)
    output_video = os.path.join(vid_dir, vid_name + '_FOMO_SAHI_448.avi')
    output_json = os.path.join(vid_dir, vid_name + '_FOMO_SAHI_448.json')

    process_video(video_path, output_video, output_json, model, frame_interval=1)


if __name__ == "__main__":
    main()