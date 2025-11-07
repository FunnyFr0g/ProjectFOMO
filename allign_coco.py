import json
from label_pathes import *


def simple_align_coco(reference_json, prediction_json, output_json):
    """
    Простая функция для выравнивания одного файла предсказаний
    """
    # Загружаем данные
    with open(reference_json, 'r') as f:
        ref_data = json.load(f)

    with open(prediction_json, 'r') as f:
        pred_data = json.load(f)

    # Создаем маппинг filename -> image_id из эталона
    ref_mapping = {img['file_name']: img['id'] for img in ref_data['images']}

    # Обновляем изображения и создаем маппинг старых ID -> новых ID
    new_images = []
    old_to_new_id = {}  # Маппинг старых image_id на новые

    for img in pred_data['images']:
        filename = img['file_name']
        if filename in ref_mapping:
            old_id = img['id']  # Сохраняем старый ID
            new_id = ref_mapping[filename]  # Получаем новый ID из эталона
            img['id'] = new_id  # Обновляем ID изображения
            new_images.append(img)
            old_to_new_id[old_id] = new_id  # Сохраняем маппинг

    # Обновляем аннотации используя созданный маппинг
    new_annotations = []
    if 'annotations' in pred_data:
        for ann in pred_data['annotations']:
            old_image_id = ann['image_id']
            if old_image_id in old_to_new_id:
                new_ann = ann.copy()
                new_ann['image_id'] = old_to_new_id[old_image_id]
                new_annotations.append(new_ann)
            else:
                print(f"Предупреждение: аннотация с image_id {old_image_id} не найдена в маппинге")

    # Сохраняем результат
    aligned_data = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': pred_data.get('categories', [])
    }

    with open(output_json, 'w') as f:
        json.dump(aligned_data, f, indent=2)

    print(f"Файл сохранен: {output_json}")
    print(f"Обработано: {len(new_images)} изображений, {len(new_annotations)} аннотаций")
    print(f"Создано маппингов: {len(old_to_new_id)}")

    # Дополнительная диагностика
    if 'annotations' in pred_data:
        original_ann_count = len(pred_data['annotations'])
        new_ann_count = len(new_annotations)
        if original_ann_count != new_ann_count:
            print(f"Предупреждение: потеряно аннотаций: {original_ann_count - new_ann_count}")


# Пример использования упрощенной версии
if __name__ == "__main__":
    simple_align_coco(
        reference_json=gt_pathes['drones_only_val'],
        prediction_json=r'predictions\NORESIZE_drones_only!BEST_FOMO_56_crossEntropy_dronesOnly_104e_model_weights\annotations\predictions.json',
        output_json=r'predictions\NORESIZE_drones_only!BEST_FOMO_56_crossEntropy_dronesOnly_104e_model_weights\annotations\predictions_aligned.json'
    )