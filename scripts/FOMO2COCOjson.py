import json


def remap_image_ids(gt_json_path, pred_json_path, output_path):
    # Загружаем GT файл
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)

    # Загружаем predictions файл
    with open(pred_json_path, 'r') as f:
        pred_data = json.load(f)

    # Проверяем формат pred_data
    if isinstance(pred_data, str):
        try:
            pred_data = json.loads(pred_data)  # Попробуем распарсить строку как JSON
        except json.JSONDecodeError:
            raise ValueError("PRED_FILE содержит невалидный JSON")

    # Проверяем, что pred_data - это список
    if not isinstance(pred_data, list):
        raise ValueError("PRED_FILE должен содержать список предсказаний")

    # Создаем словарь для маппинга: ключ - текущий image_id, значение - новый image_id
    image_id_mapping = {}
    for idx, image in enumerate(gt_data['images'], start=1):
        image_id_mapping[image['id']] = idx

    # Обновляем image_id в predictions
    for pred in pred_data:
        if not isinstance(pred, dict):
            print(f"Warning: пропускаем элемент неправильного формата: {pred}")
            continue

        old_id = pred.get('image_id')
        if old_id is None:
            print(f"Warning: пропускаем предсказание без image_id: {pred}")
            continue

        if old_id in image_id_mapping:
            pred['image_id'] = image_id_mapping[old_id]
        else:
            print(f"Warning: image_id {old_id} not found in GT file")

    # Сохраняем исправленные predictions
    with open(output_path, 'w') as f:
        json.dump(pred_data, f, indent=2)

    print(f"Remapped predictions saved to {output_path}")


if __name__ == "__main__":
    from label_pathes import gt_pathes, pred_pathes
    # Укажите пути к вашим файлам здесь
    GT_FILE = gt_pathes['mva23_val_FOMO']
    PRED_FILE = r'FOMO_50e_predictions.json'
    OUTPUT_FILE = 'FOMO_50e_predictions_UPDATE.json'

    remap_image_ids(GT_FILE, PRED_FILE, OUTPUT_FILE)

