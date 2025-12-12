import cv2
import torch
import argparse
import os
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from collections import defaultdict


class BirdTrackerAdvanced:
    def __init__(self, model_path='yolov8n.pt', tracker_type='botsort', conf_threshold=0.3, iou_threshold=0.5):
        """
        Инициализация трекера с поддержкой BoT-SORT и ByteTrack

        Args:
            model_path: путь к модели YOLOv8
            tracker_type: 'botsort' или 'bytetrack'
            conf_threshold: порог уверенности детекции
            iou_threshold: порог для NMS
        """
        # Загружаем модель YOLOv8
        self.model = YOLO(model_path)

        # Настройки трекера
        self.tracker_type = tracker_type.lower()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Создаем конфигурацию трекера
        self.tracker_config = self._create_tracker_config()

        # Для сохранения истории треков
        self.tracks_history = defaultdict(list)
        self.frame_counter = 0

    def _create_tracker_config(self):
        """Создание конфигурационного файла для трекера"""
        if self.tracker_type == 'botsort':
            tracker_config = """
            tracker_type: botsort
            track_high_thresh: 0.3
            track_low_thresh: 0.1
            new_track_thresh: 0.4
            track_buffer: 30
            match_thresh: 0.8
            fuse_score: True
            """
        elif self.tracker_type == 'bytetrack':
            tracker_config = """
            tracker_type: bytetrack
            track_high_thresh: 0.5
            track_low_thresh: 0.1
            new_track_thresh: 0.6
            track_buffer: 30
            match_thresh: 0.8
            fuse_score: True
            """
        else:
            raise ValueError(f"Unknown tracker type: {self.tracker_type}. Use 'botsort' or 'bytetrack'")

        return tracker_config

    def detect_and_track(self, video_path, output_path='tracked_video.mp4',
                         save_labels=True, labels_dir='labels', classes_of_interest=[14]):
        """
        Обработка видео с детекцией и трекингом

        Args:
            video_path: путь к входному видео
            output_path: путь для сохранения видео с треками
            save_labels: сохранять ли метки в файл
            labels_dir: директория для сохранения меток
            classes_of_interest: список интересующих классов (14 - bird в COCO)
        """
        # Открываем видео
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть видео {video_path}")
            return None

        # Получаем параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Создаем VideoWriter для сохранения результата
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Создаем директорию для меток если нужно
        if save_labels:
            os.makedirs(labels_dir, exist_ok=True)

        # Для сохранения меток
        labels_data = defaultdict(list)

        print(f"Начинаю обработку видео: {video_path}")
        print(f"Размер: {width}x{height}, FPS: {fps}, Всего кадров: {total_frames}")
        print(f"Используется трекер: {self.tracker_type}")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Конвертируем BGR в RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Выполняем трекинг с помощью YOLOv8
            results = self.model.track(
                frame_rgb,
                persist=True,  # важный параметр для трекинга между кадрами
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=classes_of_interest,  # фильтруем только птиц
                tracker=self.tracker_config,
                verbose=False
            )

            # Получаем результаты трекинга
            tracked_objects = []

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes

                for box, track_id, cls, conf in zip(boxes.xyxy, boxes.id, boxes.cls, boxes.conf):
                    x1, y1, x2, y2 = map(int, box[:4])
                    track_id = int(track_id)
                    cls = int(cls)
                    conf = float(conf)

                    # Фильтруем по классу (птицы)
                    if cls in classes_of_interest:
                        tracked_objects.append({
                            'id': track_id,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': cls
                        })

                        # Сохраняем в историю
                        self.tracks_history[track_id].append({
                            'frame': frame_count,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf
                        })

                        # Сохраняем для экспорта
                        labels_data[track_id].append({
                            'frame': frame_count,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': cls
                        })

            # Визуализируем результаты
            frame_with_tracks = self._visualize_tracks(frame, tracked_objects)

            # Запись кадра
            out.write(frame_with_tracks)

            # Прогресс
            if frame_count % 100 == 0:
                print(f"Обработано кадров: {frame_count}/{total_frames}, "
                      f"Треков: {len(tracked_objects)}")

            frame_count += 1

            # Показ видео в реальном времени (опционально)
            cv2.imshow('Bird Tracking', frame_with_tracks)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Закрываем все
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Сохраняем метки
        if save_labels:
            self._save_labels(labels_data, labels_dir, video_path)

        print(f"\nОбработка завершена!")
        print(f"Видео сохранено как: {output_path}")
        print(f"Обнаружено уникальных птиц: {len(labels_data)}")

        return labels_data

    def _visualize_tracks(self, frame, tracked_objects):
        """Визуализация треков на кадре"""
        frame_copy = frame.copy()

        # Цвета для разных ID
        colors = [
            (0, 255, 0),  # зеленый
            (255, 0, 0),  # синий
            (0, 0, 255),  # красный
            (255, 255, 0),  # голубой
            (0, 255, 255),  # желтый
            (255, 0, 255),  # розовый
            (128, 0, 128),  # фиолетовый
            (0, 128, 128),  # оливковый
            (255, 165, 0),  # оранжевый
            (128, 128, 0),  # оливковый
        ]

        # Рисуем траектории из истории
        for track_id, history in self.tracks_history.items():
            if len(history) > 1:
                points = []
                for i in range(max(0, len(history) - 30), len(history)):  # последние 30 точек
                    bbox = history[i]['bbox']
                    center_x = (bbox[0] + bbox[2]) // 2
                    center_y = (bbox[1] + bbox[3]) // 2
                    points.append((center_x, center_y))

                # Рисуем траекторию
                for i in range(1, len(points)):
                    color = colors[track_id % len(colors)]
                    cv2.line(frame_copy, points[i - 1], points[i], color, 2, cv2.LINE_AA)

        # Рисуем текущие bounding boxes
        for obj in tracked_objects:
            track_id = obj['id']
            x1, y1, x2, y2 = obj['bbox']
            confidence = obj['confidence']

            # Выбираем цвет по ID
            color = colors[track_id % len(colors)]

            # Рисуем bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

            # Заполненный прямоугольник для текста
            label = f"ID:{track_id} {confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            cv2.rectangle(frame_copy,
                          (x1, y1 - text_height - 10),
                          (x1 + text_width, y1),
                          color, -1)

            # Текст
            cv2.putText(frame_copy, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Рисуем центр объекта
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame_copy, (center_x, center_y), 3, color, -1)

        # Информация о кадре
        info_text = f"Frame: {self.frame_counter} | Birds: {len(tracked_objects)}"
        cv2.putText(frame_copy, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame_copy, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        self.frame_counter += 1
        return frame_copy

    def _save_labels(self, labels_data, labels_dir, video_path):
        """Сохранение меток в JSON и CSV файлы"""
        video_name = Path(video_path).stem

        # Сохраняем в JSON
        json_file = os.path.join(labels_dir, f"{video_name}_{self.tracker_type}_labels.json")

        serializable_data = {}
        for track_id, frames in labels_data.items():
            serializable_data[str(track_id)] = frames

        with open(json_file, 'w') as f:
            json.dump(serializable_data, f, indent=2, default=str)

        print(f"Метки JSON сохранены в: {json_file}")

        # Сохраняем в CSV
        csv_file = os.path.join(labels_dir, f"{video_name}_{self.tracker_type}_labels.csv")
        with open(csv_file, 'w') as f:
            f.write("track_id,frame,x1,y1,x2,y2,confidence,class\n")
            for track_id, frames in labels_data.items():
                for frame_data in frames:
                    bbox = frame_data['bbox']
                    f.write(f"{track_id},{frame_data['frame']},"
                            f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},"
                            f"{frame_data['confidence']},{frame_data['class']}\n")

        print(f"Метки CSV сохранены в: {csv_file}")

        # Сохраняем сводную статистику
        stats_file = os.path.join(labels_dir, f"{video_name}_{self.tracker_type}_stats.txt")
        self._save_statistics(labels_data, stats_file)

    def _save_statistics(self, labels_data, stats_file):
        """Сохранение статистики по трекам"""
        with open(stats_file, 'w') as f:
            f.write("Статистика трекинга птиц\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Трекер: {self.tracker_type.upper()}\n")
            f.write(f"Порог уверенности: {self.conf_threshold}\n")
            f.write(f"Порог IoU: {self.iou_threshold}\n\n")

            f.write("Информация по трекам:\n")
            f.write("-" * 40 + "\n")

            total_tracks = len(labels_data)
            total_detections = sum(len(frames) for frames in labels_data.values())

            f.write(f"Всего уникальных треков: {total_tracks}\n")
            f.write(f"Всего детекций: {total_detections}\n\n")

            for track_id, frames in labels_data.items():
                if len(frames) > 0:
                    frame_numbers = [f['frame'] for f in frames]
                    start_frame = min(frame_numbers)
                    end_frame = max(frame_numbers)
                    duration = end_frame - start_frame
                    avg_confidence = np.mean([f['confidence'] for f in frames])

                    f.write(f"Трек ID {track_id}:\n")
                    f.write(f"  Количество детекций: {len(frames)}\n")
                    f.write(f"  Начальный кадр: {start_frame}\n")
                    f.write(f"  Конечный кадр: {end_frame}\n")
                    f.write(f"  Длительность (кадры): {duration}\n")
                    f.write(f"  Средняя уверенность: {avg_confidence:.3f}\n")
                    f.write(f"  Первая позиция: {frames[0]['bbox']}\n")
                    f.write(f"  Последняя позиция: {frames[-1]['bbox']}\n\n")


def main():
    parser = argparse.ArgumentParser(description='Детекция и трекинг птиц с использованием YOLOv8 и BoT-SORT/ByteTrack')
    parser.add_argument('--video', type=str, required=True,
                        help='Путь к входному видео')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Путь к модели YOLOv8 (по умолчанию: yolov8n.pt)')
    parser.add_argument('--tracker', type=str, default='botsort',
                        choices=['botsort', 'bytetrack'],
                        help='Тип трекера: botsort или bytetrack (по умолчанию: botsort)')
    parser.add_argument('--output', type=str, default='tracked_video.mp4',
                        help='Путь для сохранения видео с треками')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Порог уверенности детекции (по умолчанию: 0.3)')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='Порог IoU для NMS (по умолчанию: 0.5)')
    parser.add_argument('--labels-dir', type=str, default='labels',
                        help='Директория для сохранения меток')

    args = parser.parse_args()

    # Проверяем наличие модели
    if not os.path.exists(args.model):
        print(f"Модель {args.model} не найдена. Загружаю YOLOv8n...")
        # Модель будет автоматически загружена

    # Инициализация трекера
    tracker = BirdTrackerAdvanced(
        model_path=args.model,
        tracker_type=args.tracker,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )

    # Запуск детекции и трекинга
    labels_data = tracker.detect_and_track(
        video_path=args.video,
        output_path=args.output,
        save_labels=True,
        labels_dir=args.labels_dir,
        classes_of_interest=[14]  # 14 - bird в COCO
    )


if __name__ == "__main__":
    main()