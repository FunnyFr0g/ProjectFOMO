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
from clearml import Task

# from imgs.script.qwe import output_file

# task = Task()


# Константы
BOX_SIZE = 8  # Размер стороны квадратного bounding box'а в пикселях
TRUNK_AT = 4
NUM_CLASSES = 2
W, H = 1920, 1080
VIDEO_SIZE = (H, W)
MODEL_NAME = 'FOMO_104e'

SHOW_IMAGE = False

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
    orig_h, orig_w = frame.shape[:2]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(VIDEO_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    frame_tensor = transform(frame)
    return frame_tensor.unsqueeze(0), (orig_w, orig_h)


def scale_coords(coords, from_size=(56, 56), to_size=(224, 224), orig_size=None):
    """Масштабирование координат из feature map в оригинальное разрешение"""
    if orig_size is None:
        orig_size = to_size

    orig_size = orig_size[::-1]

    # Сначала масштабируем к входному размеру модели (224x224)
    y_scale = to_size[0] / from_size[0]
    x_scale = to_size[1] / from_size[1]

    # Затем масштабируем к оригинальному разрешению изображения
    y_scale *= orig_size[0] / to_size[0]
    x_scale *= orig_size[1] / to_size[1]

    scaled_coords = []
    for y, x in coords:
        scaled_coords.append((int(y * y_scale), int(x * x_scale)))
    return scaled_coords


def process_predictions(pred_mask, pred_probs, orig_size, image_id):
    """Обработка предсказаний с добавлением confidence score и объединением близких пикселей"""
    annotations = []
    annotation_id = 1

    kernel = np.ones((3, 3), np.uint8)

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

            scaled_coords = scale_coords([(y_centroid, x_centroid)],
                                         from_size=pred_mask.shape,
                                         to_size=VIDEO_SIZE,
                                         orig_size=orig_size)
            y_orig, x_orig = scaled_coords[0]

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


def process_video(video_path, output_video, output_json, model, frame_interval=1):
    """Обработка видеофайла с сохранением результатов детекции"""
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    if SHOW_IMAGE:
        cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)

    all_annotations = []
    all_images = []
    frame_count = 0
    image_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        if frame_count % 60 == 0:
            print(f"Processing frame {frame_count} из {total_frames} progress {frame_count/total_frames*100:.2f}%")

        image_id += 1
        frame_tensor, orig_size = prepare_frame(frame)

        with torch.no_grad():
            output = model(frame_tensor)
            probs = F.softmax(output, dim=1).squeeze(0).cpu().numpy()
            # print(probs.shape)
            # int(input('probs'))
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            # print(pred_mask.shape)
            # int(input('pred_mask'))

        # Создаем запись о кадре
        image_info = {
            "id": image_id,
            "file_name": f"frame_{frame_count:06d}.jpg",
            "width": orig_size[0],
            "height": orig_size[1],
            "frame_number": frame_count
        }
        all_images.append(image_info)

        # Обрабатываем предсказания
        annotations = process_predictions(pred_mask, probs, orig_size, image_id)
        all_annotations.extend(annotations)

        # Визуализация (опционально)
        visualize_detections(frame, annotations, frame_count, output_video, mask = probs[1])

        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break


    cap.release()

    # Сохраняем результаты
    save_to_coco_format(all_images, all_annotations, output_json)
    print(f"Predictions saved to {output_json}")


def visualize_detections(frame, annotations, frame_count, output_video, mask=None):
    """Визуализация детекций на кадре"""
    vis_frame = frame.copy()

    if mask is not None and SHOW_IMAGE:
        # print(mask.shape)
        mask_cv = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.imshow('Mask', mask_cv)

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
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


    # Текст про разрешение
    cv2.putText(vis_frame, str(VIDEO_SIZE[::-1]), (0, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)

    # Показываем или сохраняем результат
    if SHOW_IMAGE:
        cv2.imshow('Detections', vis_frame)

    # Для сохранения результатов в видео файл (опционально)
    if not hasattr(visualize_detections, 'out_video'):
        h, w = vis_frame.shape[:2]
        if output_video.endswith('mp4'):
            codec = 'mp4v'
        else:
            codec = 'XVID'

        fourcc = cv2.VideoWriter_fourcc(*codec)
        fps = 60
        visualize_detections.out_video = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    visualize_detections.out_video.write(vis_frame)


# Основной код
def main():
    model = FomoModel(num_classes=NUM_CLASSES)
    checkpoint_path = 'weights/BEST_FOMO_56_crossEntropy_dronesOnly_104e_model_weights.pth'
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Обработка видеофайла
    vid_dir = r'Data/2025-06-05 14-31-20/'
    vid_name = '2025-06-05 14-31-20.mp4'
    video_path = os.path.join(vid_dir, vid_name)
    output_video = os.path.join(vid_dir, vid_name + f'_{MODEL_NAME}_{VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}.mp4')
    output_json = os.path.join(vid_dir, vid_name + f'_{MODEL_NAME}_{VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}.json')

    # video_path = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\vid2\vid2_1s.mp4'  # Укажите путь к вашему видеофайлу
    # output_json = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\vid2\vid2_1s.json'
    # output_video = r"X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\vid2\vid2_1s_FOMO.mp4"

    process_video(video_path, output_video, output_json, model, frame_interval=1)

    # Закрываем все окна OpenCV
    cv2.destroyAllWindows()
    if hasattr(visualize_detections, 'out_video'):
        visualize_detections.out_video.release()


if __name__ == "__main__":
    main()