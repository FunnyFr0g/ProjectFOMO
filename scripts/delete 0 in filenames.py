import json


def update_coco_data(coco_json_path, output_json_path):
    '''Изменяет 00011.jpg -> 11.jpg (Потому что YOLO так делает автоматом)
    и меняет category_id: 0 -> 1, Потому что в COCO нумерация должна быть 1'''


    # Загружаем COCO JSON файл
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # 1. Обновляем имена файлов в разделе images
    for image in coco_data['images']:
        old_filename = image['file_name']
        # Удаляем ведущие нули и .jpg
        number = old_filename.split('.')[0].lstrip('0')
        # Формируем новое имя файла
        new_filename = f"{number}.jpg"
        image['file_name'] = new_filename

    # 2. Обновляем category_id в разделе annotations (0 → 1)
    for annotation in coco_data['annotations']:
        if annotation['category_id'] == 0:
            annotation['category_id'] = 1

    # 3. Проверяем и обновляем categories, если нужно
    # (если есть категория с id=0, меняем её на id=1)
    for category in coco_data['categories']:
        if category['id'] == 0:
            category['id'] = 1

    # Сохраняем обновленный JSON
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"Updated COCO JSON saved to {output_json_path}")


coco_json = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\annotations\split_val_coco.json'
updated_coco_json = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\mva23_val_UPDATED.json'


update_coco_data(coco_json, updated_coco_json)