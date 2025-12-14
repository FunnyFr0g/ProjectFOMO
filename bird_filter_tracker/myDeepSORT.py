import torch
import torch.nn as nn
import numpy as np
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.track import Track
from deep_sort.sort.kalman_filter import KalmanFilter
from collections import deque


class MobileNetV2Embedder:
    def __init__(self, model_path, input_size=224, device='cuda'):
        """
        Инициализация эмбеддера на основе MobileNetV2

        Args:
            model_path: путь к сохраненной модели
            input_size: размер входного изображения
            device: устройство для вычислений
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Предобработка изображений (такая же как при обучении)
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path):
        """Загрузка предобученной модели"""
        # Предполагаем, что модель имеет два выхода: эмбеддинги и классы
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        return model

    def preprocess(self, image_crop):
        """Предобработка одного кадрированного изображения"""
        if isinstance(image_crop, np.ndarray):
            image_crop = Image.fromarray(image_crop)

        image_tensor = self.transform(image_crop).unsqueeze(0)
        return image_tensor

    def __call__(self, crops):
        """
        Получение эмбеддингов и классификация для списка кадров

        Args:
            crops: список кадрированных изображений (bounding boxes)

        Returns:
            embeddings: эмбеддинги размером (N, embedding_dim)
            class_probs: вероятности классов (N, 2) где [bird_prob, drone_prob]
        """
        if len(crops) == 0:
            return np.array([]), np.array([])

        # Батчовая обработка
        batch_tensors = []
        for crop in crops:
            tensor = self.preprocess(crop)
            batch_tensors.append(tensor)

        batch = torch.cat(batch_tensors, dim=0).to(self.device)

        with torch.no_grad():
            # Предполагаем, что модель возвращает кортеж (embeddings, class_output)
            embeddings, class_output = self.model(batch)

            # Получаем вероятности классов
            class_probs = torch.softmax(class_output, dim=1)

        return embeddings.cpu().numpy(), class_probs.cpu().numpy()


class FilteredTrack(Track):
    """Расширенный класс трека с фильтрацией птиц"""

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 embedding=None, n_history=5, bird_threshold=0.7):
        super().__init__(mean, covariance, track_id, n_init, max_age,
                         embedding=embedding)

        # История классификаций для фильтрации птиц
        self.class_history = deque(maxlen=n_history)
        self.n_history = n_history
        self.bird_threshold = bird_threshold

        # Статус фильтрации
        self.is_filtered = False

    def update_classification(self, class_probs):
        """
        Обновление истории классификации и проверка на птицу

        Args:
            class_probs: вероятности классов [bird_prob, drone_prob]
        """
        self.class_history.append(class_probs)

        if len(self.class_history) >= self.n_history:
            # Усредняем вероятность птицы по истории
            avg_bird_prob = np.mean([prob[0] for prob in self.class_history])

            # Фильтруем если средняя вероятность птицы выше порога
            self.is_filtered = avg_bird_prob > self.bird_threshold
        else:
            # Пока недостаточно истории, не фильтруем
            self.is_filtered = False

    def is_drone(self):
        """Проверка, является ли трек дроном (не отфильтрован)"""
        return not self.is_filtered


class CustomDeepSORT(DeepSort):
    """Кастомный DeepSORT с MobileNetV2 эмбеддером и фильтрацией птиц"""

    def __init__(self, model_path, input_size=224, n_history=5, bird_threshold=0.7, **kwargs):
        """
        Args:
            model_path: путь к модели MobileNetV2
            input_size: размер входного изображения
            n_history: количество последних детекций для усреднения
            bird_threshold: порог фильтрации птиц
        """
        # Инициализируем эмбеддер
        self.embedder = MobileNetV2Embedder(model_path, input_size)

        # Сохраняем параметры фильтрации
        self.n_history = n_history
        self.bird_threshold = bird_threshold

        # Вызываем родительский конструктор с кастомными параметрами
        super().__init__(**kwargs)

        # Переопределяем трекер с нашим фильтрованным треком
        self.tracker = self.create_tracker()

    def create_tracker(self):
        """Создание трекера с кастомными треками"""
        kalman_filter = KalmanFilter()

        return FilteredTracker(
            metric=self.metric,
            max_iou_distance=self.max_iou_distance,
            max_age=self.max_age,
            n_init=self.n_init,
            kalman_filter=kalman_filter,
            n_history=self.n_history,
            bird_threshold=self.bird_threshold
        )

    def update(self, bbox_xyxy, confidences, ori_img):
        """
        Обновление трекера с фильтрацией птиц

        Args:
            bbox_xyxy: боксы в формате [x1, y1, x2, y2]
            confidences: уверенности детекций
            ori_img: оригинальное изображение

        Returns:
            filtered_outputs: только треки дронов
        """
        # Получаем кадры для каждого бокса
        crops = []
        for bbox in bbox_xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            crop = ori_img[y1:y2, x1:x2]
            crops.append(crop)

        # Получаем эмбеддинги и классификации
        embeddings, class_probs = self.embedder(crops)

        # Обновляем трекер (родительский метод)
        outputs = super().update(bbox_xyxy, confidences, ori_img,
                                 embeddings=embeddings)

        # Фильтруем только дронов
        filtered_outputs = []
        for output in outputs:
            track_id = output[-1]
            track = self.tracker.tracks[track_id]

            # Обновляем классификацию для трека
            idx = outputs.index(output)
            track.update_classification(class_probs[idx])

            # Добавляем только если это дрон
            if track.is_drone():
                filtered_outputs.append(output)

        return filtered_outputs


class FilteredTracker:
    """Кастомный трекер с поддержкой FilteredTrack"""

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3,
                 kalman_filter=None, n_history=5, bird_threshold=0.7):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.kalman_filter = kalman_filter
        self.n_history = n_history
        self.bird_threshold = bird_threshold

        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Предсказание следующего состояния для всех треков"""
        for track in self.tracks:
            track.predict(self.kalman_filter)

    def update(self, detections):
        """Обновление треков с детекциями"""
        # Реализация аналогична оригинальному DeepSORT
        # но с использованием FilteredTrack

        # Сопоставление, предсказание и обновление
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Обновление совпавших треков
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kalman_filter, detections[detection_idx])

        # Создание новых треков для несопоставленных детекций
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        # Удаление старых треков
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def _initiate_track(self, detection):
        """Инициализация нового трека"""
        mean, covariance = self.kalman_filter.initiate(detection.to_xyah())

        track = FilteredTrack(
            mean, covariance,
            track_id=self._next_id,
            n_init=self.n_init,
            max_age=self.max_age,
            embedding=detection.feature,
            n_history=self.n_history,
            bird_threshold=self.bird_threshold
        )

        self.tracks.append(track)
        self._next_id += 1


# Пример использования
def main():
    # Параметры
    model_path = "path/to/your/mobilenet_v2_model.pth"

    # Инициализация кастомного DeepSORT
    tracker = CustomDeepSORT(
        model_path=model_path,
        input_size=224,  # или тот размер, на котором обучали
        n_history=5,
        bird_threshold=0.7,
        max_dist=0.2,  # максимальное расстояние для ассоциации
        max_iou_distance=0.7,
        max_age=30,
        n_init=3,
        nn_budget=100
    )

    # Пример обработки кадра
    def process_frame(frame, detections):
        """
        Обработка одного кадра

        Args:
            frame: изображение (numpy array)
            detections: детекции в формате [[x1, y1, x2, y2, conf], ...]
        """
        if len(detections) == 0:
            return []

        # Разделяем детекции
        bbox_xyxy = np.array([d[:4] for d in detections])
        confidences = np.array([d[4] for d in detections])

        # Обновляем трекер
        tracks = tracker.update(bbox_xyxy, confidences, frame)

        # Возвращаем только треки дронов
        return tracks


if __name__ == "__main__":
    main()