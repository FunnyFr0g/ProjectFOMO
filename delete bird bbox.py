import json
import os
import shutil


def create_dataset_with_background_images(original_json, new_json,
                                          original_images_dir, new_images_dir,
                                          keyword='bird'):
    """
    Простая версия: сохраняет все изображения, но аннотации только для keyword изображений
    """

    # Загружаем исходные данные
    with open(original_json, 'r') as f:
        data = json.load(f)

    # Создаем выходные директории
    os.makedirs(new_images_dir, exist_ok=True)
    os.makedirs(os.path.dirname(new_json) if os.path.dirname(new_json) else '.', exist_ok=True)

    # Находим ID изображений с ключевым словом
    keyword_ids = set()
    for image in data['images']:
        if keyword.lower() in image['file_name'].lower():
            keyword_ids.add(image['id'])

        # Копируем все изображения
        src = os.path.join(original_images_dir, image['file_name'])
        dst = os.path.join(new_images_dir, image['file_name'])
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # Фильтруем аннотации
    filtered_annotations = [
        ann for ann in data['annotations']
        if ann['image_id'] in keyword_ids
    ]

    # Создаем новые данные
    new_data = {
        'images': data['images'],  # Все изображения
        'annotations': filtered_annotations,  # Только аннотации для keyword
        'categories': data['categories'],
        'info': data.get('info', {}),
        'licenses': data.get('licenses', [])
    }

    # Сохраняем
    with open(new_json, 'w') as f:
        json.dump(new_data, f, indent=2)

    print(f"✅ Датсет создан!")
    print(f"📁 Изображений: {len(data['images'])}")
    print(f"🎯 С '{keyword}': {len(keyword_ids)}")
    print(f"📦 Аннотаций: {len(filtered_annotations)}")
    print(f"💾 Путь: {new_images_dir}")


# Использование
create_dataset_with_background_images(
    original_json=r"X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test/skb_test.json",
    new_json=r"X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test_bg\annotations.json",
    original_images_dir=r"X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test\images",
    new_images_dir=r"X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test_bg\images",
    keyword='drone'
)
