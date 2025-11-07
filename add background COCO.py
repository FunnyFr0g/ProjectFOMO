import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def add_background_images_to_coco(original_coco_json_path,  # Путь к исходному JSON COCO
                                  background_images_dir,     # Папка с фоновыми изображениями
                                  output_coco_json_path,     # Куда сохранить новый JSON
                                  output_images_dir,         # Куда копировать фоновые изображения
                                  dataset_type='train'):     # Для какого набора (train/val)
    """
    Добавляет фоновые изображения в аннотации COCO.

    Args:
        original_coco_json_path (str): Путь к исходному файлу аннотаций COCO.
        background_images_dir (str): Папка с фоновыми изображениями.
        output_coco_json_path (str): Путь для сохранения нового файла аннотаций.
        output_images_dir (str): Папка, в которую будут скопированы фоновые изображения.
        dataset_type (str): Тип датасета (например, 'train', 'val'). Добавляется к имени файла.
    """

    # Загрузка исходного COCO JSON
    with open(original_coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Создание папки для выходных изображений, если её нет
    os.makedirs(output_images_dir, exist_ok=True)

    # Получение максимального существующего image_id
    if coco_data['images']:
        max_image_id = max([img['id'] for img in coco_data['images']])
    else:
        max_image_id = 0
    print(f"Текущий максимальный image_id: {max_image_id}")

    # Поддерживаемые форматы изображений
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # Проход по всем фоновым изображениям
    background_images = [f for f in os.listdir(background_images_dir) if f.lower().endswith(valid_extensions)]
    print(f"Найдено {len(background_images)} фоновых изображений.")

    new_images_list = []

    for i, image_file in tqdm(enumerate(background_images)):
        # Генерация нового уникального image_id
        new_image_id = max_image_id + i + 1

        # Полный путь к исходному фоновому изображению
        src_image_path = os.path.join(background_images_dir, image_file)

        # Создание нового имени для файла (опционально, но рекомендуется для избежания конфликтов)
        # Например, можно добавить префикс 'bg_'
        # new_filename = f"bg_{dataset_type}_{image_file}"
        new_filename = image_file
        dst_image_path = os.path.join(output_images_dir, new_filename)

        # Копирование изображения в целевую папку
        shutil.copy2(src_image_path, dst_image_path)
        print(f"Скопировано: {image_file} -> {new_filename}")

        # Создание записи для изображения в формате COCO
        image_info = {
            "id": new_image_id,
            "file_name": new_filename,  # Важно: использовать новое имя файла
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        }

        from PIL import Image
        with Image.open(src_image_path) as img:
            width, height = img.size
        image_info["width"] = width
        image_info["height"] = height

        new_images_list.append(image_info)

    coco_data['images'].extend(new_images_list)

    # Важно: Раздел 'annotations' НЕ изменяется. На эти изображения нет аннотаций.

    # Сохранение обновленного JSON
    with open(output_coco_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2) # indent для читаемости

    print(f"Готово! Новый файл аннотаций сохранен: {output_coco_json_path}")
    print(f"Добавлено {len(new_images_list)} фоновых изображений.")
    print(f"Общее количество изображений в датасете: {len(coco_data['images'])}")
    print(f"Общее количество аннотаций в датасете: {len(coco_data['annotations'])}")

if __name__ == "__main__":
    # Пути к вашим данным
    path_to_original_annotations = r"C:\Users\ILYA\.clearml\cache\storage_manager\datasets\dronesonly_FOMO_bg_crop\train\train_annotations.json"
    path_to_backgrounds = r"Data/background_crop"
    path_to_new_annotations = r"C:\Users\ILYA\.clearml\cache\storage_manager\datasets\dronesonly_FOMO_bg_crop\train\train_annotations_bg.json"
    path_to_output_images = r"C:\Users\ILYA\.clearml\cache\storage_manager\datasets\dronesonly_FOMO_bg_crop\train\images" # Та же папка, где лежат оригинальные изображения

    add_background_images_to_coco(
        original_coco_json_path=path_to_original_annotations,
        background_images_dir=path_to_backgrounds,
        output_coco_json_path=path_to_new_annotations,
        output_images_dir=path_to_output_images,
        dataset_type='train'
    )