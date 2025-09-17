import cv2
import os
import numpy as np

iou = 0.1

def combine_images(folder_path, output_path=fr'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\ROC\combined_{iou=}.jpg'):
    # Получаем список всех файлов в папке
    files = os.listdir(folder_path)

    def correct_order(name:str):
        if '640' in name:
            return 0
        elif '1088' in name:
            return 1
        elif 'baseline' in name:
            return 2
        return -1
    s_iou = str(iou)

    # Фильтруем изображения для первой строки (содержат "mva" и "0.1")
    line1_images = [f for f in files if 'mva' in f.lower() and s_iou in f]
    print(f'{line1_images = }')
    line1_images.sort(key=correct_order)
    print(f'sorted{line1_images = } ')

    # Фильтруем изображения для второй строки (содержат "vid1" и "0.1")
    line2_images = [f for f in files if 'vid1' in f.lower() and s_iou in f and ('1088px2' not in f.lower())]
    print(f'{line2_images = }')
    line2_images.sort(key=correct_order)
    print(f'sorted{line2_images = } ')

    # Фильтруем изображения для второй строки (содержат "vid1" и "0.1")
    line3_images = [f for f in files if 'skb_test' in f.lower() and s_iou in f]
    print(f'{line3_images = }')
    line3_images.sort(key=correct_order)
    print(f'sorted{line3_images = } ')


    # Загружаем изображения для первой строки
    imgs_line1 = []
    for img_path in line1_images:
        img = cv2.imread(os.path.join(folder_path, img_path))
        if img is not None:
            imgs_line1.append(img)

    # Загружаем изображения для второй строки
    imgs_line2 = []
    for img_path in line2_images:
        img = cv2.imread(os.path.join(folder_path, img_path))
        if img is not None:
            imgs_line2.append(img)

    imgs_line3 = []
    for img_path in line3_images:
        img = cv2.imread(os.path.join(folder_path, img_path))
        if img is not None:
            imgs_line3.append(img)

    # Если нет изображений - выходим
    if not imgs_line1 and not imgs_line2:
        print("Не найдено подходящих изображений")
        return

    # Склеиваем изображения по горизонтали для каждой строки
    line1 = np.hstack(imgs_line1) if imgs_line1 else None
    line2 = np.hstack(imgs_line2) if imgs_line2 else None
    line3 = np.hstack(imgs_line3) if imgs_line3 else None

    # Склеиваем строки по вертикали
    if line1 is not None and line2 is not None:
        combined = np.vstack((line1, line2, line3))
    elif line1 is not None:
        combined = line1
    else:
        combined = line2

    # Сохраняем результат
    cv2.imwrite(output_path, combined)
    print(f"Изображения успешно склеены и сохранены как {output_path}")


# Укажите путь к вашей папке с изображениями
folder_path = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\ROC'
combine_images(folder_path)