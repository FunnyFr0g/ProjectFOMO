import os
import json
import cv2
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from torchvision.models import mobilenet_v2

# Константы
BOX_SIZE = 16
TRUNK_AT = 4
NUM_CLASSES = 2
W, H = 4 * 224, 4 * 224
VIDEO_SIZE = (H, W)
MODEL_NAME = 'FOMO_22e_bg_cr'

SHOW_IMAGE = True

params = {
    "NUM_CLASSES": 2,
    'DO_RESIZE': True,
    "INPUT_SIZE": VIDEO_SIZE,
    "BOX_SIZE": BOX_SIZE,
}


# ==================== SequenceEncoder (точная копия из вашего кода обучения) ====================
class SequenceEncoder(nn.Module):
    def __init__(self, embedding_dim=512, dropout_rate=0.3):
        super().__init__()

        # Backbone - MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=False)

        # Заменяем классификатор
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

        # Временный модуль для агрегации последовательности
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # Классификатор для птиц/дронов (опционально)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)  # 2 класса: птица, дрон
        )

    def forward(self, x, return_classification=False):
        """
        x: тензор формы (batch_size, sequence_length, 3, H, W)
        """
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Обрабатываем каждый кадр отдельно
        frame_embeddings = []
        for i in range(seq_len):
            # (batch_size, 3, H, W)
            frame = x[:, i, :, :, :]

            # Получаем эмбеддинг для каждого кадра
            embedding = self.backbone(frame)  # (batch_size, embedding_dim)
            frame_embeddings.append(embedding)

        # Собираем все эмбеддинги
        # (batch_size, seq_len, embedding_dim)
        all_embeddings = torch.stack(frame_embeddings, dim=1)

        # Агрегируем по временной оси (берем среднее)
        sequence_embedding = torch.mean(all_embeddings, dim=1)  # (batch_size, embedding_dim)

        # Нормализуем эмбеддинг (важно для triplet loss)
        sequence_embedding = F.normalize(sequence_embedding, p=2, dim=1)

        if return_classification:
            # Предсказание класса (птица/дрон)
            class_logits = self.classifier(sequence_embedding)
            return sequence_embedding, class_logits

        return sequence_embedding

    def extract_features(self, x):
        """Только извлечение признаков без классификации"""
        return self.forward(x, return_classification=False)


# ==================== MobileNetV2 Embedder адаптированный для SequenceEncoder ====================
class MobileNetV2Embedder:
    def __init__(self, model_path, input_size=32, embedding_dim=512, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size

        # Загружаем модель SequenceEncoder
        self.model = SequenceEncoder(embedding_dim=embedding_dim, dropout_rate=0.3)

        # Загружаем веса
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Если сохранена вся модель
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        # Трансформации ДОЛЖНЫ совпадать с теми, что использовались при обучении!
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, crops):
        """Извлекаем эмбеддинги и вероятности классов"""
        if len(crops) == 0:
            return np.array([]), np.array([])

        batch_tensors = []
        for crop in crops:
            if isinstance(crop, np.ndarray):
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = Image.fromarray(crop)

            # Применяем трансформации
            tensor = self.transform(crop)  # (3, H, W)

            # Добавляем dimension для последовательности (последовательность длины 1)
            # SequenceEncoder ожидает (batch_size, sequence_length, 3, H, W)
            tensor = tensor.unsqueeze(0)  # (1, 3, H, W) - sequence_length=1
            batch_tensors.append(tensor)

        # Создаем батч: (N, 1, 3, H, W) где N = len(crops)
        batch = torch.stack(batch_tensors, dim=0)  # (N, 1, 3, H, W)
        batch = batch.to(self.device)

        with torch.no_grad():
            # Получаем эмбеддинги и логиты классов
            embeddings, class_logits = self.model(batch, return_classification=True)

            # Преобразуем логиты в вероятности
            class_probs = torch.softmax(class_logits, dim=1)

        return embeddings.cpu().numpy(), class_probs.cpu().numpy()


# ==================== FOMO Model (оставляем без изменений) ====================
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


# ==================== Simple Tracker with Bird Filtering ====================
class Track:
    def __init__(self, track_id, bbox, embedding, class_prob, n_history=5, bird_threshold=0.7):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.embedding = embedding
        self.class_history = deque(maxlen=n_history)
        self.class_history.append(class_prob)
        self.n_history = n_history
        self.bird_threshold = bird_threshold
        self.age = 0
        self.hits = 1
        self.time_since_update = 0

    def update(self, bbox, embedding, class_prob):
        self.bbox = bbox
        self.embedding = embedding
        self.class_history.append(class_prob)
        self.hits += 1
        self.time_since_update = 0

    def predict(self):
        self.time_since_update += 1

    def is_drone(self):
        if len(self.class_history) == 0:
            return True

        # Средняя вероятность птицы по истории
        # class_prob имеет форму [bird_prob, drone_prob]
        avg_bird_prob = np.mean([prob[0] for prob in self.class_history])
        return avg_bird_prob < self.bird_threshold

    def get_color(self):
        if self.is_drone():
            return (0, 255, 0)  # Зеленый для дрона
        else:
            return (0, 0, 255)  # Красный для птицы


class SimpleTracker:
    def __init__(self, max_dist=0.5, max_age=30, n_history=5, bird_threshold=0.7):
        self.max_dist = max_dist
        self.max_age = max_age
        self.n_history = n_history
        self.bird_threshold = bird_threshold
        self.tracks = []
        self.next_id = 1

    def cosine_distance(self, emb1, emb2):
        """Косинусное расстояние между эмбеддингами"""
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
        return 1 - np.dot(emb1, emb2)

    def iou_distance(self, bbox1, bbox2):
        """IoU расстояние между боксами"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 < x1 or y2 < y1:
            return 1.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return 1 - (intersection / (union + 1e-8))

    def update(self, detections, embeddings, class_probs):
        """
        Обновление трекеров с новыми детекциями

        Args:
            detections: список боксов [[x1, y1, x2, y2], ...]
            embeddings: эмбеддинги для каждого бокса
            class_probs: вероятности классов для каждого бокса [bird_prob, drone_prob]
        """
        # Предсказание для существующих треков
        for track in self.tracks:
            track.predict()

        if len(detections) == 0:
            # Удаляем старые треки
            self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
            return self.tracks

        # Матрица расстояний (IoU + Cosine)
        n_tracks = len(self.tracks)
        n_dets = len(detections)

        if n_tracks == 0:
            # Создаем новые треки для всех детекций
            for i in range(n_dets):
                track = Track(self.next_id, detections[i], embeddings[i],
                              class_probs[i], self.n_history, self.bird_threshold)
                self.tracks.append(track)
                self.next_id += 1
            return self.tracks

        cost_matrix = np.zeros((n_tracks, n_dets))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                # Комбинированное расстояние
                iou_dist = self.iou_distance(track.bbox, det)
                cos_dist = self.cosine_distance(track.embedding, embeddings[j])
                cost_matrix[i, j] = 0.5 * iou_dist + 0.5 * cos_dist

        # Простое greedy matching
        matches = []
        unmatched_tracks = list(range(n_tracks))
        unmatched_detections = list(range(n_dets))

        for i in range(n_tracks):
            if i not in unmatched_tracks:
                continue

            min_cost = 1.0
            best_match = -1

            for j in range(n_dets):
                if j not in unmatched_detections:
                    continue

                if cost_matrix[i, j] < min_cost and cost_matrix[i, j] < self.max_dist:
                    min_cost = cost_matrix[i, j]
                    best_match = j

            if best_match != -1:
                matches.append((i, best_match))
                unmatched_tracks.remove(i)
                unmatched_detections.remove(best_match)

        # Обновляем совпавшие треки
        for i, j in matches:
            self.tracks[i].update(detections[j], embeddings[j], class_probs[j])

        # Создаем новые треки для несопоставленных детекций
        for j in unmatched_detections:
            track = Track(self.next_id, detections[j], embeddings[j],
                          class_probs[j], self.n_history, self.bird_threshold)
            self.tracks.append(track)
            self.next_id += 1

        # Удаляем старые треки
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]

        return self.tracks


# ==================== Utility Functions (оставляем без изменений) ====================
if torch.cuda.is_available():
    MEAN_GPU = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    STD_GPU = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
else:
    MEAN_CPU = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD_CPU = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def prepare_image(image, new_size=params["INPUT_SIZE"], to_cuda=True, to_cpu=False):
    """Подготовка изображения для FOMO модели"""
    orig_h, orig_w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if params['DO_RESIZE']:
        image = cv2.resize(image, new_size)

    image = np.ascontiguousarray(image)
    tensor = torch.from_numpy(image).float()
    tensor = tensor.permute(2, 0, 1).div_(255.0)

    if to_cuda and torch.cuda.is_available():
        tensor = tensor.cuda()
        tensor = tensor.unsqueeze(0)
        tensor = (tensor - MEAN_GPU) / STD_GPU
    else:
        tensor = tensor.unsqueeze(0)
        tensor = (tensor - MEAN_CPU) / STD_CPU

    if to_cpu:
        tensor = tensor.cpu()

    return tensor, (orig_w, orig_h)


def scale_coords(coords, from_size=(56, 56), to_size=(224, 224), orig_size=None):
    """Масштабирование координат"""
    if orig_size is None:
        orig_size = to_size

    orig_size = orig_size[::-1]  # Важный фикс для неквадратного изображения

    y_scale = to_size[0] / from_size[0]
    x_scale = to_size[1] / from_size[1]

    y_scale *= orig_size[0] / to_size[0]
    x_scale *= orig_size[1] / to_size[1]

    scaled_coords = []
    for y, x in coords:
        scaled_coords.append((int(y * y_scale), int(x * x_scale)))
    return scaled_coords


def process_predictions(pred_mask, pred_probs, orig_size, image_id):
    """Обработка предсказаний FOMO"""
    annotations = []
    annotation_id = 1

    for class_id in range(1, NUM_CLASSES):
        class_mask = (pred_mask == class_id).astype(np.uint8)

        if len(class_mask.shape) > 2:
            class_mask = class_mask.squeeze()

        y_coords, x_coords = np.where(class_mask == 1)

        if len(y_coords) == 0:
            continue

        points = list(zip(y_coords, x_coords))
        scores = pred_probs[class_id][y_coords, x_coords]

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
    """Сохранение в COCO формате"""
    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "drone"}],
        "info": {"description": "FOMO + Tracking with bird filtering"},
        "licenses": [{"id": 1, "name": "CC-BY"}]
    }

    with open(output_path, 'w') as f:
        json.dump(coco_output, f, indent=2)


# ==================== Main Processing Function ====================
def process_video(video_path, output_video, output_json, fomo_model, mobilenet_model,
                  frame_interval=1, n_history=5, bird_threshold=0.7):
    """Обработка видео с трекингом и фильтрацией птиц"""
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    if SHOW_IMAGE:
        cv2.namedWindow("Tracking with Bird Filtering", cv2.WINDOW_NORMAL)

    # Инициализация трекера
    tracker = SimpleTracker(max_dist=0.5, max_age=30,
                            n_history=n_history, bird_threshold=bird_threshold)

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
            print(f"Processing frame {frame_count} из {total_frames} progress {frame_count / total_frames * 100:.2f}%")

        image_id += 1
        orig_size = (frame.shape[1], frame.shape[0])

        # Детекция FOMO
        frame_tensor, _ = prepare_image(frame)

        with torch.no_grad():
            output = fomo_model(frame_tensor)
            probs = F.softmax(output, dim=1).squeeze(0).cpu().numpy()
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # Получение аннотаций от FOMO
        fomo_annotations = process_predictions(pred_mask, probs, orig_size, image_id)

        # Подготовка детекций для трекинга
        detections = []
        crops = []

        for ann in fomo_annotations:
            x1, y1, w, h = ann["bbox"]
            x2, y2 = x1 + w, y1 + h

            # Вырезаем регион с запасом
            pad = 5
            x1_pad = max(0, int(x1) - pad)
            y1_pad = max(0, int(y1) - pad)
            x2_pad = min(frame.shape[1], int(x2) + pad)
            y2_pad = min(frame.shape[0], int(y2) + pad)

            if (x2_pad - x1_pad) > 5 and (y2_pad - y1_pad) > 5:
                crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                crops.append(crop)
                detections.append([x1, y1, x2, y2])

        # Извлечение признаков MobileNetV2 (SequenceEncoder)
        if len(crops) > 0:
            embeddings, class_probs = mobilenet_model.extract_features(crops)
        else:
            embeddings, class_probs = np.array([]), np.array([])

        # Обновление трекера
        if len(detections) > 0 and len(embeddings) > 0:
            tracks = tracker.update(detections, embeddings, class_probs)
        else:
            tracks = tracker.update([], [], [])

        # Создание записи о кадре
        image_info = {
            "id": image_id,
            "file_name": f"frame_{frame_count:06d}.jpg",
            "width": orig_size[0],
            "height": orig_size[1],
            "frame_number": frame_count
        }
        all_images.append(image_info)

        # Создание аннотаций только для дронов
        for track in tracks:
            if track.is_drone() and track.hits > 3:  # Только подтвержденные треки дронов
                x1, y1, x2, y2 = track.bbox
                w, h = x2 - x1, y2 - y1

                annotation = {
                    "id": track.track_id,
                    "image_id": image_id,
                    "category_id": 1,  # Дрон
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "area": float(w * h),
                    "score": float(np.mean([prob[1] for prob in track.class_history])),  # Вероятность дрона
                    "iscrowd": 0,
                    "centroid": [float(x1 + w / 2), float(y1 + h / 2)]
                }
                all_annotations.append(annotation)

        # Визуализация
        vis_frame = frame.copy()

        # Рисуем все треки
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            color = track.get_color()

            # Прямоугольник
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

            # ID трека
            label = f"ID: {track.track_id}"
            cv2.putText(vis_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Информация о классификации
            if len(track.class_history) > 0:
                avg_bird_prob = np.mean([prob[0] for prob in track.class_history])
                avg_drone_prob = 1.0 - avg_bird_prob
                class_text = f"DRONE: {(avg_drone_prob):.2f}" if track.is_drone() else f"BIRD: {avg_bird_prob:.2f}"
                cv2.putText(vis_frame, class_text, (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Статистика
        drones_count = sum(1 for t in tracks if t.is_drone())
        birds_count = len(tracks) - drones_count
        stats_text = f"Drones: {drones_count}, Birds: {birds_count}"
        cv2.putText(vis_frame, stats_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Показ кадра
        if SHOW_IMAGE:
            cv2.imshow('Tracking with Bird Filtering', vis_frame)

        # Сохранение в видео
        if not hasattr(process_video, 'out_video'):
            h, w = vis_frame.shape[:2]
            if output_video.endswith('mp4'):
                codec = 'mp4v'
            else:
                codec = 'XVID'

            fourcc = cv2.VideoWriter_fourcc(*codec)
            fps = 30 if cap.get(cv2.CAP_PROP_FPS) > 30 else cap.get(cv2.CAP_PROP_FPS)
            process_video.out_video = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

        process_video.out_video.write(vis_frame)

        # Выход по ESC
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break

    cap.release()

    # Сохранение результатов
    save_to_coco_format(all_images, all_annotations, output_json)
    print(f"Predictions saved to {output_json}")

    if hasattr(process_video, 'out_video'):
        process_video.out_video.release()


# ==================== Main Function ====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка FOMO модели
    fomo_model = FomoModel(num_classes=NUM_CLASSES)
    fomo_checkpoint_path = 'weights/FOMO_56_bg_crop_drones_only_FOMO_1.0.2/BEST_22e.pth'
    fomo_model.load_state_dict(torch.load(fomo_checkpoint_path, map_location=device))
    fomo_model = fomo_model.to(device)
    fomo_model.eval()

    # Загрузка MobileNetV2 модели (SequenceEncoder) для классификации
    mobilenet_checkpoint_path = 'weights/mobilenet_encoder 32/final_model.pth'  # Укажите путь к вашей обученной модели
    mobilenet_model = MobileNetV2Embedder(
        model_path=mobilenet_checkpoint_path,
        input_size=32,  # Размер изображения при обучении
        embedding_dim=512,  # Размерность эмбеддингов при обучении
        device=device
    )

    # Обработка видео
    vid_dir = r'G:\SOD vid'
    vid_name = 'vid_4.mp4'
    video_path = os.path.join(vid_dir, vid_name)
    output_video = os.path.join(vid_dir, vid_name + f'_tracking_{MODEL_NAME}_myDeepSORT_bsz={BOX_SIZE}.mp4')
    output_json = os.path.join(vid_dir, vid_name + f'_tracking_{MODEL_NAME}_myDeepSORT_bsz={BOX_SIZE}.json')

    process_video(
        video_path,
        output_video,
        output_json,
        fomo_model,
        mobilenet_model,
        frame_interval=1,
        n_history=5,  # История для усреднения
        bird_threshold=0.85  # Порог фильтрации птиц
    )

    cv2.destroyAllWindows()
    print("Processing completed!")


if __name__ == "__main__":
    main()