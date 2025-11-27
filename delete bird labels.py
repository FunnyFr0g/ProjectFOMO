import json
import os
import shutil


def create_keyword_dataset(input_json, output_json, images_dir, output_images_dir, keyword='bird'):
    """
    Создает новый датасет COCO только с изображениями, содержащими ключевое слово в имени файла.
    Остальные изображения игнорируются.
    """

    # Загружаем аннотации
    with open(input_json, 'r') as f:
        data = json.load(f)

    # Создаем выходные директории
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    # Фильтруем изображения по ключевому слову
    keyword_image_ids = set()
    filtered_images = []

    for image in data['images']:
        if keyword.lower() in image['file_name'].lower():
            keyword_image_ids.add(image['id'])
            filtered_images.append(image)

            # Копируем изображение
            src = os.path.join(images_dir, image['file_name'])
            dst = os.path.join(output_images_dir, image['file_name'])
            if os.path.exists(src):
                shutil.copy2(src, dst)

    # Фильтруем аннотации только для выбранных изображений
    filtered_annotations = [
        ann for ann in data['annotations']
        if ann['image_id'] in keyword_image_ids
    ]

    # Создаем новый JSON
    new_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': data['categories'],
        'info': data.get('info', {}),
        'licenses': data.get('licenses', [])
    }

    # Сохраняем новый JSON
    with open(output_json, 'w') as f:
        json.dump(new_data, f, indent=2)

    print(f"Создан датасет с ключевым словом '{keyword}':")
    print(f"- Изображений: {len(filtered_images)}")
    print(f"- Аннотаций: {len(filtered_annotations)}")
    print(f"- Путь к изображениям: {output_images_dir}")
    print(f"- Путь к аннотациям: {output_json}")


# Использование
create_keyword_dataset(
    input_json=r"X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test/skb_test.json",
    output_json="X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test_nobird\images\skb_test_nobird.json",
    images_dir=r"X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test\images",
    output_images_dir="X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test_nobird\images",
    keyword='drone'
)