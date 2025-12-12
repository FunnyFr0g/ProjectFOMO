# combined_bird_detection_pipeline.py
import cv2
import torch
import numpy as np
from pathlib import Path
import time
from collections import deque
from ultralytics import YOLO
import supervision as sv
from torchvision.models import mobilenet_v2
import torch.nn as nn
import torch.nn.functional as F
import json

MIN_FRAMES_HISTORY = 2
INPUT_SIZE = (224, 224)

class FomoBackbone(nn.Module):
    def __init__(self, trunk_at=4):
        super().__init__()
        self.mobilenet = mobilenet_v2(pretrained=False).features[:trunk_at]

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
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = FomoBackbone()
        self.head = FomoHead(num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class FOMODetector:
    def __init__(self, model_path, box_size=8, num_classes=2, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.box_size = box_size
        self.num_classes = num_classes

        # Загрузка модели FOMO
        self.model = FomoModel(num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        # Константы для нормализации
        if self.device.type == 'cuda':
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        else:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        # Размеры
        self.input_size = INPUT_SIZE #(448,448)#(224, 224) #(896, 896)  # 4*224
        self.feature_map_size = [x//4 for x in self.input_size]  # После mobilenet (уменьшение в 4 раза)

    def prepare_image(self, image, new_size=None):
        """Подготовка изображения для FOMO"""
        if new_size is None:
            new_size = self.input_size

        # BGR → RGB и ресайз
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, new_size)

        # Конвертация в тензор
        image = np.ascontiguousarray(image)
        tensor = torch.from_numpy(image).float()

        # Реорганизация и нормализация
        tensor = tensor.permute(2, 0, 1).div_(255.0)  # CHW и [0,1]
        tensor = tensor.unsqueeze(0).to(self.device)  # Добавляем batch dimension
        tensor = (tensor - self.mean) / self.std

        return tensor, (image.shape[1], image.shape[0])  # Возвращаем (width, height)

    def scale_coords(self, coords, from_size, to_size, orig_size):
        """Масштабирование координат из feature map в оригинальное разрешение"""
        orig_h, orig_w = orig_size
        to_h, to_w = to_size

        # Масштаб от feature map к input size
        y_scale_input = to_h / from_size[0]
        x_scale_input = to_w / from_size[1]

        # Масштаб от input size к оригинальному размеру
        y_scale_orig = orig_h / to_h
        x_scale_orig = orig_w / to_w

        scaled_coords = []
        for y, x in coords:
            y_scaled = y * y_scale_input * y_scale_orig
            x_scaled = x * x_scale_input * x_scale_orig
            scaled_coords.append((int(y_scaled), int(x_scaled)))
        return scaled_coords

    def detect(self, frame):
        """Детекция с помощью FOMO"""
        orig_h, orig_w = frame.shape[:2]

        # Подготовка изображения
        frame_tensor, _ = self.prepare_image(frame)

        with torch.no_grad():
            output = self.model(frame_tensor)
            probs = F.softmax(output, dim=1).squeeze(0).cpu().numpy()
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # Обработка предсказаний (только класс 1 - птицы)
        class_id = 1
        class_mask = (pred_mask == class_id).astype(np.uint8)

        if len(class_mask.shape) > 2:
            class_mask = class_mask.squeeze()

        # Находим ненулевые пиксели
        y_coords, x_coords = np.where(class_mask == 1)

        if len(y_coords) == 0:
            return [], [], []

        # Создаем список точек и соответствующих scores
        points = list(zip(y_coords, x_coords))
        scores = probs[class_id][y_coords, x_coords]

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

        bboxes = []
        confidences = []

        for group in groups:
            if not group:
                continue

            group_points = [points[i] for i in group]
            group_scores = [scores[i] for i in group]

            y_coords_group = [p[0] for p in group_points]
            x_coords_group = [p[1] for p in group_points]

            # Вычисляем центроид группы
            y_centroid = np.mean(y_coords_group)
            x_centroid = np.mean(x_coords_group)
            score = np.mean(group_scores)

            # Масштабируем центроид к оригинальному разрешению
            scaled_coords = self.scale_coords(
                [(y_centroid, x_centroid)],
                from_size=pred_mask.shape,  # (56, 56)
                to_size=self.input_size,  # (896, 896)
                orig_size=(orig_h, orig_w)
            )

            y_orig, x_orig = scaled_coords[0]

            # Создаем bounding box фиксированного размера вокруг центроида
            half_size = self.box_size // 2
            x1 = max(0, int(x_orig - half_size))
            y1 = max(0, int(y_orig - half_size))
            x2 = min(orig_w, int(x_orig + half_size))
            y2 = min(orig_h, int(y_orig + half_size))

            # Проверяем, что bbox имеет положительные размеры
            if x2 > x1 and y2 > y1:
                bboxes.append([x1, y1, x2, y2])
                confidences.append(float(score))

        return np.array(bboxes), np.array(confidences), np.array([])  # track_ids пустые


class CombinedBirdVerificationSystem:
    def __init__(self, predictor_path, encoder_path,
                 yolo_model='yolo11s.pt', fomo_model_path=None,
                 use_fomo=False, fusion_strategy='union'):
        """
        Комбинированная система верификации с поддержкой YOLO и FOMO

        Args:
            predictor_path: Путь к весам предсказательной модели
            encoder_path: Путь к весам энкодера
            yolo_model: Модель YOLO
            fomo_model_path: Путь к весам FOMO модели
            use_fomo: Использовать ли FOMO для детекции
            fusion_strategy: Стратегия слияния детекций ('union', 'priority_yolo', 'priority_fomo')
        """
        self.use_fomo = use_fomo
        self.fusion_strategy = fusion_strategy

        # Инициализация детекторов
        if not use_fomo:
            # Только YOLO
            print(f"Загрузка YOLO модели: {yolo_model}")
            self.model_yolo = YOLO(yolo_model)
            self.detector = self.detect_with_yolo
        else:
            if fomo_model_path is None:
                raise ValueError("Не указан путь к модели FOMO")

            # Загрузка FOMO
            print("Загрузка модели FOMO...")
            self.fomo_detector = FOMODetector(fomo_model_path, box_size=32)
            self.detector = self.detect_with_fomo

        # Параметры детекции
        self.conf_threshold = 0.1
        self.iou_threshold = 0.5
        self.classes_to_detect = [0, 1, 14]  # COCO: bird
        self.imgsz = 1088

        # Загрузка наших моделей для верификации
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используемое устройство: {self.device}")

        # Предсказательная модель
        from pf_convlstm_predictor import ConvLSTMPredictor
        self.predictor = ConvLSTMPredictor(input_channels=3, hidden_dims=[32, 64])
        predictor_checkpoint = torch.load(predictor_path, map_location=self.device)
        self.predictor.load_state_dict(predictor_checkpoint['model_state_dict'])
        self.predictor.to(self.device)
        self.predictor.eval()

        # Энкодер
        from pf_encoder_trainer import BirdEncoder
        self.encoder = BirdEncoder(embedding_dim=128, pretrained=False)
        encoder_checkpoint = torch.load(encoder_path, map_location=self.device)
        self.encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
        self.encoder.to(self.device)
        self.encoder.eval()

        # Трекинг и история
        self.track_history = {}  # track_id -> deque of frames
        self.hidden_states = {}  # track_id -> hidden states for predictor
        self.frame_buffer = {}  # Буфер для smooth трекинга

        # Параметры
        self.sequence_length = 5
        self.img_size = 32
        self.similarity_threshold = 0.5

        # Инициализация трекера
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.05,
            minimum_consecutive_frames=3,
            minimum_matching_threshold=0.5
        )

        # Статистика
        self.stats = {
            'total_detections': 0,
            'verified_detections': 0,
            'similarity_scores': [],
            'yolo_detections': 0,
            'fomo_detections': 0,
            'fused_detections': 0
        }

    def detect_with_yolo(self, frame):
        """Детекция с YOLO"""
        results = self.model_yolo(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes_to_detect,
            verbose=False,
            imgsz=self.imgsz,
        )[0]

        if results.boxes is None:
            return [], [], []

        # Конвертация в формат supervision
        detections = sv.Detections.from_ultralytics(results)

        # Трекинг
        detections = self.tracker.update_with_detections(detections)

        # Фильтрация по классу (птицы)
        bird_indices = []
        for i, class_id in enumerate(detections.class_id):
            if class_id in [0, 1, 14, 16]:  # COCO bird classes
                bird_indices.append(i)

        if not bird_indices:
            return [], [], []

        # Отбираем только детекции птиц
        bird_detections = detections[bird_indices]

        # Извлекаем информацию
        bboxes = bird_detections.xyxy
        confidences = bird_detections.confidence
        track_ids = bird_detections.tracker_id

        return bboxes, confidences, track_ids

    def detect_with_fomo(self, frame):
        """Детекция с FOMO"""
        bboxes, confidences, _ = self.fomo_detector.detect(frame)

        if len(bboxes) == 0:
            return [], [], []

        # Конвертация в формат supervision
        detections = sv.Detections(
            xyxy=bboxes,
            confidence=confidences,
            class_id=np.ones(len(bboxes), dtype=int)  # Все детекции - птицы (класс 1)
        )

        # Трекинг
        detections = self.tracker.update_with_detections(detections)

        return detections.xyxy, detections.confidence, detections.tracker_id

    def detect_combined(self, frame):
        """Комбинированная детекция YOLO + FOMO"""
        # Детекция YOLO
        yolo_bboxes, yolo_confs, yolo_track_ids = self.detect_with_yolo(frame)
        self.stats['yolo_detections'] = len(yolo_bboxes)

        # Детекция FOMO
        fomo_bboxes, fomo_confs, fomo_track_ids = self.detect_with_fomo(frame)
        self.stats['fomo_detections'] = len(fomo_bboxes)

        if self.fusion_strategy == 'union':
            # Объединение детекций (удаление дубликатов по IoU)
            return self.fuse_detections_union(yolo_bboxes, yolo_confs, fomo_bboxes, fomo_confs)
        elif self.fusion_strategy == 'priority_yolo':
            # Приоритет YOLO, FOMO только для пропущенных
            return self.fuse_detections_priority(yolo_bboxes, yolo_confs, fomo_bboxes, fomo_confs, priority='yolo')
        elif self.fusion_strategy == 'priority_fomo':
            # Приоритет FOMO, YOLO только для подтверждения
            return self.fuse_detections_priority(yolo_bboxes, yolo_confs, fomo_bboxes, fomo_confs, priority='fomo')
        else:
            return yolo_bboxes, yolo_confs, yolo_track_ids

    def fuse_detections_union(self, bboxes1, confs1, bboxes2, confs2, iou_threshold=0.3):
        """Объединение детекций с удалением дубликатов"""
        if len(bboxes1) == 0:
            return bboxes2, confs2, []
        if len(bboxes2) == 0:
            return bboxes1, confs1, []

        all_bboxes = np.vstack([bboxes1, bboxes2])
        all_confs = np.concatenate([confs1, confs2])

        # Сортируем по confidence
        sorted_indices = np.argsort(all_confs)[::-1]
        all_bboxes = all_bboxes[sorted_indices]
        all_confs = all_confs[sorted_indices]

        # NMS для удаления дубликатов
        keep_indices = []
        used = [False] * len(all_bboxes)

        for i in range(len(all_bboxes)):
            if used[i]:
                continue

            keep_indices.append(i)
            used[i] = True

            for j in range(i + 1, len(all_bboxes)):
                if used[j]:
                    continue

                # Вычисляем IoU
                iou = self.calculate_iou(all_bboxes[i], all_bboxes[j])
                if iou > iou_threshold:
                    used[j] = True

        filtered_bboxes = all_bboxes[keep_indices]
        filtered_confs = all_confs[keep_indices]

        # Конвертация в формат supervision для трекинга
        detections = sv.Detections(
            xyxy=filtered_bboxes,
            confidence=filtered_confs,
            class_id=np.ones(len(filtered_bboxes), dtype=int)
        )
        detections = self.tracker.update_with_detections(detections)

        self.stats['fused_detections'] = len(filtered_bboxes)
        return detections.xyxy, detections.confidence, detections.tracker_id

    def fuse_detections_priority(self, bboxes1, confs1, bboxes2, confs2, priority='yolo', iou_threshold=0.3):
        """Слияние с приоритетом одного детектора"""
        if priority == 'yolo':
            primary_bboxes, primary_confs = bboxes1, confs1
            secondary_bboxes, secondary_confs = bboxes2, confs2
        else:
            primary_bboxes, primary_confs = bboxes2, confs2
            secondary_bboxes, secondary_confs = bboxes1, confs1

        if len(primary_bboxes) == 0:
            return secondary_bboxes, secondary_confs, []

        # Помечаем secondary детекции, которые не пересекаются с primary
        keep_secondary = []
        for i, sec_bbox in enumerate(secondary_bboxes):
            overlap = False
            for prim_bbox in primary_bboxes:
                iou = self.calculate_iou(sec_bbox, prim_bbox)
                if iou > iou_threshold:
                    overlap = True
                    break

            if not overlap:
                keep_secondary.append(i)

        # Объединяем
        if len(keep_secondary) > 0:
            all_bboxes = np.vstack([primary_bboxes, secondary_bboxes[keep_secondary]])
            all_confs = np.concatenate([primary_confs, secondary_confs[keep_secondary]])
        else:
            all_bboxes = primary_bboxes
            all_confs = primary_confs

        # Конвертация в формат supervision для трекинга
        detections = sv.Detections(
            xyxy=all_bboxes,
            confidence=all_confs,
            class_id=np.ones(len(all_bboxes), dtype=int)
        )
        detections = self.tracker.update_with_detections(detections)

        self.stats['fused_detections'] = len(all_bboxes)
        return detections.xyxy, detections.confidence, detections.tracker_id

    def calculate_iou(self, bbox1, bbox2):
        """Вычисление IoU между двумя bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        iou = inter_area / float(bbox1_area + bbox2_area - inter_area + 1e-6)
        return iou

    def preprocess_bbox(self, frame, bbox):
        """Вырезает и нормализует bounding box с учетом паддинга"""
        x1, y1, x2, y2 = map(int, bbox)

        # Добавляем паддинг (минимум 5 пикселей)
        width = x2 - x1
        height = y2 - y1
        pad_x = max(int(width * 0.1), 5)
        pad_y = max(int(height * 0.1), 5)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame.shape[1], x2 + pad_x)
        y2 = min(frame.shape[0], y2 + pad_y)

        cropped = frame[y1:y2, x1:x2]

        if cropped.size == 0:
            return None

        # Сохраняем пропорции
        h, w = cropped.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(cropped, (new_w, new_h))

        # Создаем квадратное изображение с паддингом
        square = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        y_offset = (self.img_size - new_h) // 2
        x_offset = (self.img_size - new_w) // 2
        square[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        # Преобразование для модели
        square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        square = square.astype(np.float32) / 255.0

        # В тензор
        tensor = torch.from_numpy(square).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device), (x1, y1, x2, y2)

    def update_track_history(self, track_id, frame_tensor, bbox_info):
        """Обновляет историю кадров для трека"""
        if track_id not in self.track_history:
            self.track_history[track_id] = {
                'frames': deque(maxlen=self.sequence_length),
                'bboxes': deque(maxlen=self.sequence_length),
                'hidden': None,
                'last_seen': 0
            }

        track_data = self.track_history[track_id]
        track_data['frames'].append(frame_tensor)
        track_data['bboxes'].append(bbox_info)
        track_data['last_seen'] = time.time()

        # Если накопили достаточно кадров, делаем предсказание
        if len(track_data['frames']) == self.sequence_length:
            sequence = torch.stack(list(track_data['frames']))

            with torch.no_grad():
                predicted_frame, new_hidden = self.predictor(
                    sequence,
                    track_data['hidden']
                )
                track_data['hidden'] = new_hidden

            return predicted_frame[-1]

        return None

    def verify_detection(self, track_id, current_frame_tensor, bbox_info):
        """Проверяет детекцию через сравнение эмбеддингов"""
        if track_id not in self.track_history:
            return False, 0.0

        track_data = self.track_history[track_id]
        frames = track_data['frames']

        if len(frames) < MIN_FRAMES_HISTORY:  # Нужна хотя бы небольшая история
            return True, 1.0  # Пропускаем первые детекции

        # Получаем предсказанный кадр
        if len(frames) == self.sequence_length:
            sequence = torch.stack(list(frames))

            with torch.no_grad():
                predicted_frame, _ = self.predictor(sequence, track_data['hidden'])

                # Эмбеддинги
                pred_embedding = self.encoder(predicted_frame)
                curr_embedding = self.encoder(current_frame_tensor)

                # Косинусное сходство
                similarity = torch.nn.functional.cosine_similarity(
                    pred_embedding, curr_embedding, dim=1
                ).mean().item()

                # Обновляем статистику
                self.stats['similarity_scores'].append(similarity)

                # Верификация
                verified = similarity > self.similarity_threshold
                if verified:
                    self.stats['verified_detections'] += 1

                return verified, similarity

        return True, 1.0  # Если история короткая, пропускаем

    def cleanup_old_tracks(self, current_time, max_age=5.0):
        """Удаляет старые треки"""
        track_ids = list(self.track_history.keys())
        for track_id in track_ids:
            if current_time - self.track_history[track_id]['last_seen'] > max_age:
                del self.track_history[track_id]

    def process_frame(self, frame, frame_id):
        """Обрабатывает один кадр"""
        self.stats['total_detections'] = 0

        # Очистка старых треков
        self.cleanup_old_tracks(time.time())

        # Детекция
        if self.use_fomo and self.fusion_strategy != 'none':
            bboxes, confidences, track_ids = self.detect_combined(frame)
        else:
            bboxes, confidences, track_ids = self.detector(frame)

        verified_bboxes = []
        verification_scores = []

        for bbox, confidence, track_id in zip(bboxes, confidences, track_ids):
            self.stats['total_detections'] += 1

            # Предобработка bbox
            result = self.preprocess_bbox(frame, bbox)
            if result is None:
                continue

            frame_tensor, bbox_info = result

            # Обновляем историю и получаем предсказание
            predicted = self.update_track_history(track_id, frame_tensor, bbox_info)

            # Верификация
            verified, similarity = self.verify_detection(track_id, frame_tensor, bbox_info)

            if verified:
                verified_bboxes.append((bbox, track_id))
                verification_scores.append(similarity)

            # Визуализация
            detector_type = "FOMO" if self.use_fomo else "YOLO"
            color = (0, 255, 0) if verified else (0, 0, 255)
            label = f"ID:{track_id} D:{detector_type} C:{confidence:.2f} S:{similarity:.2f}"

            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Отображаем предсказание
            if predicted is not None:  # and verified:
                self.display_prediction(frame, predicted, bbox)

        # Отображаем статистику
        self.display_stats(frame, frame_id)

        return frame, verified_bboxes, verification_scores

    def display_prediction(self, frame, predicted, original_bbox):
        """Отображает предсказанный патч"""
        # Конвертируем тензор в изображение
        pred_np = predicted.squeeze().cpu().permute(1, 2, 0).numpy()
        pred_np = (pred_np * 255).astype(np.uint8)
        pred_np = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)

        # Увеличиваем для наглядности
        pred_np = cv2.resize(pred_np, (64, 64))

        # Размещаем в углу кадра
        h, w = frame.shape[:2]
        frame[h - 74:h - 10, 10:74] = pred_np

        # Рамка вокруг превью
        cv2.rectangle(frame, (8, h - 76), (76, h - 8), (255, 255, 255), 1)
        cv2.putText(frame, "Prediction", (10, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def display_stats(self, frame, frame_id):
        """Отображает статистику на кадре"""
        stats_text = [
            f"Frame: {frame_id}",
            f"Detections: {self.stats['total_detections']}",
            f"Verified: {self.stats['verified_detections']}",
            f"Threshold: {self.similarity_threshold}",
            f"Detector: {'FOMO' if self.use_fomo else 'YOLO'}"
        ]

        if self.use_fomo and self.fusion_strategy != 'none':
            stats_text.extend([
                f"YOLO: {self.stats['yolo_detections']}",
                f"FOMO: {self.stats['fomo_detections']}",
                f"Fused: {self.stats['fused_detections']}"
            ])

        if self.stats['similarity_scores']:
            avg_sim = np.mean(self.stats['similarity_scores'][-50:])
            stats_text.append(f"Avg Similarity: {avg_sim:.3f}")

        y_offset = 30
        for text in stats_text:
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25

    def run_on_video(self, video_path, output_path="output_combined.mp4"):
        """Запуск на видеофайле"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Ошибка открытия видео: {video_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Видео: {video_path}")
        print(f"Размер: {width}x{height}, FPS: {fps}, Кадров: {total_frames}")
        print(f"Детектор: {'FOMO' if self.use_fomo else 'YOLO'}")
        if self.use_fomo:
            print(f"Стратегия слияния: {self.fusion_strategy}")

        # Кодек для записи
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()
        fps_values = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Обработка кадра
            frame_start = time.time()
            processed_frame, verified, scores = self.process_frame(frame, frame_count)
            frame_time = time.time() - frame_start

            # Расчет FPS
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            fps_values.append(current_fps)

            # Запись кадра
            out.write(processed_frame)

            # Показ прогресса
            if frame_count % 30 == 0:
                avg_fps = np.mean(fps_values[-30:])
                elapsed = time.time() - start_time
                remaining = (total_frames - frame_count) / avg_fps if avg_fps > 0 else 0

                print(f"Кадр {frame_count}/{total_frames} | "
                      f"FPS: {avg_fps:.1f} | "
                      f"Вер: {self.stats['verified_detections']} | "
                      f"Время: {elapsed:.1f}s | "
                      f"Осталось: {remaining:.1f}s")

            # Показ
            display_frame = cv2.resize(processed_frame, (width // 2, height // 2))
            cv2.imshow('Bird Verification - Combined', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Остановлено пользователем")
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Итоговая статистика
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        print(f"\n{'=' * 50}")
        print(f"ОБРАБОТКА ЗАВЕРШЕНА")
        print(f"{'=' * 50}")
        print(f"Итого кадров: {frame_count}")
        print(f"Общее время: {total_time:.1f} секунд")
        print(f"Средний FPS: {avg_fps:.1f}")
        print(f"Всего детекций: {self.stats['total_detections']}")
        print(f"Верифицировано: {self.stats['verified_detections']}")

        if self.use_fomo:
            print(f"YOLO детекций: {self.stats['yolo_detections']}")
            print(f"FOMO детекций: {self.stats['fomo_detections']}")
            print(f"Объединено детекций: {self.stats['fused_detections']}")

        if self.stats['similarity_scores']:
            print(f"Среднее схожество: {np.mean(self.stats['similarity_scores']):.3f}")
            print(f"Медиана схожести: {np.median(self.stats['similarity_scores']):.3f}")

        verification_rate = (
            self.stats['verified_detections'] / self.stats['total_detections']
            if self.stats['total_detections'] > 0 else 0
        )
        print(f"Процент верификации: {verification_rate:.1%}")
        print(f"Выходной файл: {output_path}")


def main():
    """Пример использования комбинированной системы"""
    #
    # # Вариант 1: Только YOLO (как раньше)
    # print("=" * 60)
    # print("Вариант 1: Только YOLO")
    # print("=" * 60)
    #
    # system_yolo = CombinedBirdVerificationSystem(
    #     predictor_path="weights/predictor/best.pth",
    #     encoder_path="weights/encoder/best.pth",
    #     yolo_model="yolo11s.pt",
    #     use_fomo=False
    # )
    # system_yolo.run_on_video(
    #     r"G:\SOD vid\bird_3.mp4",
    #     "video/bird_3_yolo_only.mp4"
    # )
    #
    # Вариант 2: Только FOMO
    print("\n" + "=" * 60)
    print("Вариант 2: Только FOMO")
    print("=" * 60)

    system_fomo = CombinedBirdVerificationSystem(
        predictor_path="weights/predictor/best.pth",
        encoder_path="weights/encoder/best.pth",
        yolo_model="yolo11s.pt",  # Все равно загружается, но не используется
        fomo_model_path="weights/FOMO_56_bg_crop_drones_only_FOMO_1.0.2/BEST_22e.pth",
        use_fomo=True,
        fusion_strategy='none'  # Только FOMO
    )
    system_fomo.run_on_video(
        r"G:\SOD vid\vid_4.mp4",
        rf"video\vid_4_FOMO_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}_1_history.mp4"
    )

    # Вариант 3: Комбинированный (объединение)
    # print("\n" + "=" * 60)
    # print("Вариант 3: YOLO + FOMO (объединение)")
    # print("=" * 60)
    #
    # system_combined = CombinedBirdVerificationSystem(
    #     predictor_path="weights/predictor/best.pth",
    #     encoder_path="weights/encoder/best.pth",
    #     yolo_model="yolo11s.pt",
    #     fomo_model_path="weights/FOMO_56_bg_crop_drones_only_FOMO_1.0.2/BEST_22e.pth",
    #     use_fomo=True,
    #     fusion_strategy='union'
    # )
    # system_combined.run_on_video(
    #     r"Data/2025-06-05 14-31-20/2025-06-05 14-31-20.mp4",
    #     "Data/2025-06-05 14-31-20/2025-06-05 14-31-20 FOMO YOLO pipeline.mp4"
    # )
    #
    # # Вариант 4: Приоритет YOLO
    # print("\n" + "=" * 60)
    # print("Вариант 4: YOLO + FOMO (приоритет YOLO)")
    # print("=" * 60)
    #
    # system_priority_yolo = CombinedBirdVerificationSystem(
    #     predictor_path="weights/predictor/best.pth",
    #     encoder_path="weights/encoder/best.pth",
    #     yolo_model="yolo11s.pt",
    #     fomo_model_path="weights/FOMO_56_bg_crop_drones_only_FOMO_1.0.2/BEST_22e.pth",
    #     use_fomo=True,
    #     fusion_strategy='priority_yolo'
    # )
    # system_priority_yolo.run_on_video(
    #     r"G:\SOD vid\bird_3.mp4",
    #     "video/bird_3_priority_yolo.mp4"
    # )


if __name__ == "__main__":
    # Установите зависимости:
    # pip install ultralytics supervision opencv-python torch torchvision

    import os

    os.makedirs("video", exist_ok=True)

    main()