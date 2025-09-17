import cv2
import imageio

input_video = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\vid2_centernet_resnet.mp4'
output_gif = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\vid2_centernet_resnet.gif'



cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise ValueError("Не удалось открыть видео!")

fps = 60
print(f"Исходный FPS видео: {fps}")

start_frame = 13 * fps
end_frame = 16 * fps
frames = []
current_frame = 0  # Счётчик начинается с 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Видео закончилось

    # Пропускаем кадры до start_frame
    if current_frame < start_frame:
        current_frame += 1
        continue

    # Прерываем цикл, если вышли за end_frame
    if current_frame > end_frame:
        break

    # Конвертируем BGR в RGB и добавляем в список
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(rgb_frame)
    print(f"Обработан кадр {current_frame}")
    current_frame += 1

cap.release()

if not frames:
    raise ValueError("Не удалось прочитать кадры в диапазоне {start_frame}-{end_frame}!")

# Сохраняем GIF (можно регулировать FPS)
imageio.mimsave(output_gif, frames, fps=10, loop=0)
print(f"GIF сохранён: {output_gif}, кадров: {len(frames)}")