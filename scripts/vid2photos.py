import cv2
import os


def save_every_nth_frame(video_path, output_folder, n=50):
    # Создаем папку для сохранения кадров, если её нет
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Открываем видеофайл
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видеофайл")
        return

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        # Если кадр не прочитан (конец видео или ошибка)
        if not ret:
            break

        # Сохраняем каждый n-й кадр
        if frame_count % n == 0:
            output_path = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
            print(f"Сохранен кадр {frame_count}")

        frame_count += 1

    cap.release()
    print(f"Готово! Сохранено {saved_count} кадров из {frame_count}")


# Пример использования
if __name__ == "__main__":
    video_path = r"O:\Downloads\Telegram Desktop\Video_2025_06_30_193634_10.avi"  # Укажите путь к вашему видеофайлу
    output_folder = "output_frames"  # Папка для сохранения кадров

    save_every_nth_frame(video_path, output_folder, n=50)