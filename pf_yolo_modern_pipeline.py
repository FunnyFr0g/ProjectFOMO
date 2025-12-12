# yolo_modern_pipeline.py
import cv2
import torch
import numpy as np
from pathlib import Path
import time
from collections import deque
from ultralytics import YOLO
import supervision as sv


class ModernBirdVerificationSystem:
    def __init__(self, predictor_path, encoder_path, yolo_model='yolo11s.pt'):
        """
        Полная система верификации с современным YOLO

        Args:
            predictor_path: Путь к весам предсказательной модели
            encoder_path: Путь к весам энкодера
            yolo_model: Модель YOLO (yolo11n.pt, yolov8n.pt, и т.д.)
        """
        # Загрузка YOLO (современный API)
        print(f"Загрузка YOLO модели: {yolo_model}")
        self.model_yolo = YOLO(yolo_model)

        # Параметры детекции
        self.conf_threshold = 0.1
        self.iou_threshold = 0.5
        self.classes_to_detect = [0, 1, 14]  # COCO: bird
        self.imgsz = 1088

        # Загрузка наших моделей
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

        # Инициализация трекера (современный подход)
        self.tracker = sv.ByteTrack(track_activation_threshold=0.05,
                                    minimum_consecutive_frames=1,
                                    minimum_matching_threshold=0.5)
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            # text_thickness=1,
            # text_scale=0.5
        )

        # Статистика
        self.stats = {
            'total_detections': 0,
            'verified_detections': 0,
            'similarity_scores': []
        }

    def preprocess_bbox(self, frame, bbox):
        """Вырезает и нормализует bounding box с учетом паддинга"""
        x1, y1, x2, y2 = map(int, bbox)

        # Добавляем паддинг (10% от размера bbox)
        width = x2 - x1
        height = y2 - y1
        pad_x = int(width * 0.1)
        pad_y = int(height * 0.1)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame.shape[1], x2 + pad_x)
        y2 = min(frame.shape[0], y2 + pad_y)

        cropped = frame[y1:y2, x1:x2]

        if cropped.size == 0:
            return None

        # Сохраняем пропорции с паддингом
        h, w = cropped.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # resized = cv2.resize(cropped, (new_w, new_h))
        #
        # # Создаем квадратное изображение с паддингом
        # square = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        # y_offset = (self.img_size - new_h) // 2
        # x_offset = (self.img_size - new_w) // 2
        # square[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        #
        #
        # # Преобразование для модели
        # square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        # square = square.astype(np.float32) / 255.0
        square = cv2.resize(cropped, (new_w, new_h))

        # В тензор
        tensor = torch.from_numpy(square).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device), (x1, y1, x2, y2)

    def detect_with_yolo(self, frame):
        """Детекция с современным YOLO"""
        # Детекция
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
            # sequence = sequence .unsqueeze(0)[0]  # [1, T, C, H, W]

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

        if len(frames) < 2:  # Нужна хотя бы небольшая история
            return True, 1.0  # Пропускаем первые детекции

        # Получаем предсказанный кадр
        if len(frames) == self.sequence_length:
            sequence = torch.stack(list(frames))
            # sequence = sequence.unsqueeze(0)

            with torch.no_grad():
                predicted_frame, _ = self.predictor(sequence, track_data['hidden'])

                # Эмбеддинги
                pred_embedding = self.encoder(predicted_frame)
                curr_embedding = self.encoder(current_frame_tensor)

                # Косинусное сходство
                similarity = torch.nn.functional.cosine_similarity(
                    pred_embedding, curr_embedding, dim=1
                ).mean().item()  # .mean() для усреднения, .item() для скаляра

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
                if track_id in self.hidden_states:
                    del self.hidden_states[track_id]

    def process_frame(self, frame, frame_id):
        """Обрабатывает один кадр"""
        self.stats['total_detections'] = 0

        # Очистка старых треков
        self.cleanup_old_tracks(time.time())

        # Детекция YOLO
        bboxes, confidences, track_ids = self.detect_with_yolo(frame)

        verified_bboxes = []
        verification_scores = []

        for bbox, confidence, track_id in zip(bboxes, confidences, track_ids):
            self.stats['total_detections'] += 1

            # Предобработка bbox
            frame_tensor, bbox_info = self.preprocess_bbox(frame, bbox)
            if frame_tensor is None:
                continue

            # Обновляем историю и получаем предсказание
            predicted = self.update_track_history(track_id, frame_tensor, bbox_info)

            # Верификация
            verified, similarity = self.verify_detection(track_id, frame_tensor, bbox_info)

            if verified:
                verified_bboxes.append((bbox, track_id))
                verification_scores.append(similarity)

            # Визуализация
            color = (0, 255, 0) if verified else (0, 0, 255)
            label = f"ID:{track_id} C:{confidence:.2f} S:{similarity:.2f}"

            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Отображаем предсказание
            if predicted is not None: #and verified:
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
            f"Threshold: {self.similarity_threshold}"
        ]

        if self.stats['similarity_scores']:
            avg_sim = np.mean(self.stats['similarity_scores'][-50:])
            stats_text.append(f"Avg Similarity: {avg_sim:.3f}")

        y_offset = 30
        for text in stats_text:
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25

    def run_on_video(self, video_path, output_path="output_modern.mp4"):
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

        # Кодек для записи
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()

        # Для расчета FPS
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
            cv2.imshow('Bird Verification - Modern YOLO', display_frame)

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

        if self.stats['similarity_scores']:
            print(f"Среднее схожество: {np.mean(self.stats['similarity_scores']):.3f}")
            print(f"Медиана схожести: {np.median(self.stats['similarity_scores']):.3f}")

        verification_rate = (
            self.stats['verified_detections'] / self.stats['total_detections']
            if self.stats['total_detections'] > 0 else 0
        )
        print(f"Процент верификации: {verification_rate:.1%}")
        print(f"Выходной файл: {output_path}")


def evaluate_with_different_yolo_models():
    """Сравнение разных моделей YOLO"""
    models_to_test = [
        ('YOLOv11n', 'yolo11n.pt'),
        ('YOLOv8n', 'yolov8n.pt'),
        ('YOLOv8s', 'yolov8s.pt'),
        ('YOLOv11s', 'yolo11s.pt'),
    ]

    results = {}

    for model_name, model_file in models_to_test:
        print(f"\n{'=' * 60}")
        print(f"Тестирование модели: {model_name}")
        print(f"{'=' * 60}")

        try:
            # Создаем систему с текущей моделью
            system = ModernBirdVerificationSystem(
                predictor_path="best_predictor.pth",
                encoder_path="best_encoder.pth",
                yolo_model=model_file
            )

            # Тестовый прогон
            start_time = time.time()
            system.run_on_video(
                video_path="test_bird_video.mp4",
                output_path=f"output_{model_name}.mp4"
            )
            total_time = time.time() - start_time

            # Сохраняем результаты
            results[model_name] = {
                'total_time': total_time,
                'avg_fps': system.stats['total_detections'] / total_time if total_time > 0 else 0,
                'detections': system.stats['total_detections'],
                'verified': system.stats['verified_detections'],
                'avg_similarity': np.mean(system.stats['similarity_scores']) if system.stats['similarity_scores'] else 0
            }

        except Exception as e:
            print(f"Ошибка с моделью {model_name}: {e}")
            results[model_name] = {'error': str(e)}

    # Вывод сравнения
    print(f"\n{'=' * 80}")
    print(f"СРАВНЕНИЕ МОДЕЛЕЙ YOLO")
    print(f"{'=' * 80}")
    print(f"{'Модель':<15} {'FPS':<10} {'Детекций':<10} {'Вериф.':<10} {'Сходство':<10} {'Время':<10}")
    print(f"{'-' * 80}")

    for model_name, metrics in results.items():
        if 'error' in metrics:
            print(f"{model_name:<15} {'ERROR':<10} {metrics['error']}")
        else:
            print(f"{model_name:<15} {metrics['avg_fps']:<10.1f} "
                  f"{metrics['detections']:<10} {metrics['verified']:<10} "
                  f"{metrics['avg_similarity']:<10.3f} {metrics['total_time']:<10.1f}")


if __name__ == "__main__":
    # Установите зависимости:
    # pip install ultralytics supervision opencv-python

    # Вариант 1: Одна модель
    print("Запуск с YOLOv11...")
    system = ModernBirdVerificationSystem(
        predictor_path="weights/predictor/best.pth",
        encoder_path="weights/encoder/best.pth",
        yolo_model="yolo11s.pt"  # или yolov8n.pt
    )
    import os
    os.makedirs("video", exist_ok=True)
    system.run_on_video(r"G:\SOD vid\bird_3.mp4", "video/bird_3_yolo11.mp4")

    # Вариант 2: Сравнение нескольких моделей
    # evaluate_with_different_yolo_models()