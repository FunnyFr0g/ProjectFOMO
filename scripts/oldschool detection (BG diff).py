import cv2
import numpy as np


class BackgroundSubtractor:
    def __init__(self, buffer_size=5, threshold=30, min_contour_area=5, max_contour_area=50):
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        self.frame_buffer = []
        self.background = None

    def update_background(self, frame):
        """Обновление фонового изображения на основе последних N кадров"""
        if len(self.frame_buffer) >= self.buffer_size:
            self.frame_buffer.pop(0)

        # Преобразуем в оттенки серого для обработки
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_buffer.append(gray)

        if len(self.frame_buffer) == self.buffer_size:
            # Усреднение кадров для получения фона
            self.background = np.mean(self.frame_buffer, axis=0).astype(np.uint8)

    def detect_objects(self, frame):
        """Обнаружение объектов с вычитанием фона"""
        if self.background is None:
            return frame, []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Вычитание фона
        diff = cv2.absdiff(gray, self.background)

        # Пороговая обработка
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Морфологические операции для удаления шума
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Нахождение контуров
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Фильтрация контуров по площади
        filtered_contours = [cnt for cnt in contours if self.min_contour_area < cv2.contourArea(cnt) < self.max_contour_area]

        # Построение bbox'ов
        bboxes = []
        for cnt in filtered_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            bboxes.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return frame, bboxes


def main(video_path, start_frame=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка открытия видео файла")
        return

    bg_subtractor = BackgroundSubtractor(buffer_size=4, threshold=30, min_contour_area=3, max_contour_area=50)
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        print(f'{frame_index/75 :.2f}s')

        if frame_index < start_frame:
            continue


        # Обновление фона
        bg_subtractor.update_background(frame)

        # Детекция объектов
        processed_frame, bboxes = bg_subtractor.detect_objects(frame.copy())

        # Отображение результатов
        # cv2.imshow('Object Detection', processed_frame)

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, bg_subtractor.background)
            cv2.imshow('Object Detection', diff)
        except:
            cv2.imshow('Object Detection', processed_frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    fps=75
    video_path = r"O:\Downloads\Telegram Desktop\Video_2025_06_30_193634_10.avi"  # Укажите путь к вашему видео
    main(video_path, start_frame=63*fps)