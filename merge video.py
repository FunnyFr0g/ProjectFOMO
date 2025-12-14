import os

import cv2
import numpy as np
import argparse


def merge_videos(video1_path, video2_path, output_path, layout='horizontal'):
    """
    Объединяет два видео рядом или друг над другом

    Args:
        video1_path: путь к первому видео
        video2_path: путь ко второму видео
        output_path: путь для сохранения результата
        layout: 'horizontal' - рядом, 'vertical' - друг над другом
    """

    # Открываем видеофайлы
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # Проверяем, открылись ли видео
    if not cap1.isOpened():
        print(f"Ошибка: не удалось открыть видео {video1_path}")
        return
    if not cap2.isOpened():
        print(f"Ошибка: не удалось открыть видео {video2_path}")
        return

    # Получаем параметры видео
    fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
    fps2 = int(cap2.get(cv2.CAP_PROP_FPS))

    # Используем минимальный FPS для выходного видео
    fps = min(fps1, fps2)

    # Получаем размеры кадров
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Видео 1: {width1}x{height1}, {fps1} FPS")
    print(f"Видео 2: {width2}x{height2}, {fps2} FPS")

    # Определяем размеры выходного видео в зависимости от выбранного расположения
    if layout == 'horizontal':
        # Рядом: ширина суммируется, высота берется максимальная
        output_width = width1 + width2
        output_height = max(height1, height2)
    elif layout == 'vertical':
        # Друг над другом: высота суммируется, ширина берется максимальная
        output_width = max(width1, width2)
        output_height = height1 + height2
    else:
        print("Ошибка: неверный параметр layout. Используйте 'horizontal' или 'vertical'")
        return

    # Создаем объект VideoWriter для сохранения результата
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # или 'XVID' для avi
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    frame_count = 0

    print(f"Начинаю обработку. Выходное видео: {output_width}x{output_height}, {fps} FPS")

    while True:
        # Читаем кадры из обоих видео
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Если хотя бы одно видео закончилось, прерываем цикл
        if not ret1 or not ret2:
            print(f"Одно из видео закончилось. Обработано кадров: {frame_count}")
            break

        # Изменяем размеры кадров, если они разные (только если нужно)
        if layout == 'horizontal' and height1 != height2:
            # Для горизонтального расположения приводим к одинаковой высоте
            if height1 != output_height:
                frame1 = cv2.resize(frame1, (width1, output_height))
            if height2 != output_height:
                frame2 = cv2.resize(frame2, (width2, output_height))
        elif layout == 'vertical' and width1 != width2:
            # Для вертикального расположения приводим к одинаковой ширине
            if width1 != output_width:
                frame1 = cv2.resize(frame1, (output_width, height1))
            if width2 != output_width:
                frame2 = cv2.resize(frame2, (output_width, height2))

        # Объединяем кадры в зависимости от выбранного расположения
        if layout == 'horizontal':
            combined_frame = np.hstack((frame1, frame2))
        else:  # vertical
            combined_frame = np.vstack((frame1, frame2))

        # Записываем кадр в выходное видео
        out.write(combined_frame)

        # Отображаем прогресс (опционально)
        if frame_count % 100 == 0:
            print(f"Обработано кадров: {frame_count}")

        frame_count += 1

    # Освобождаем ресурсы
    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Обработка завершена! Результат сохранен в: {output_path}")




if __name__ == "__main__":
    vid1 = r'Data\2025-06-05 14-31-20\2025-06-05 14-31-20.mp4_FOMO_22e_bg_cr_896x896.mp4'
    vid2 = r'Data\2025-06-05 14-31-20\2025-06-05 14-31-20 FOMO pipeline.mp4'

    # merge_videos(vid1, vid2, 'Data/224 vs 896 vertical.mp4', layout='vertical')
    merge_videos(vid1, vid2, 'Data/default vs filter horizontal.mp4', layout='horizontal')
