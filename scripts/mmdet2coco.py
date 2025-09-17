import json
from collections import defaultdict


def convert_predictions_to_coco(predictions_file, annotations_file, output_file):
    """
    Преобразует файл предсказаний в полный COCO-формат

    :param predictions_file: Путь к файлу с предсказаниями (в вашем текущем формате)
    :param annotations_file: Путь к оригинальному COCO-файлу с аннотациями
    :param output_file: Путь для сохранения результата
    """
    # Загружаем оригинальные аннотации
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)

    # Загружаем предсказания
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    # Создаем словарь для быстрого доступа к информации об изображениях
    img_info = {img['id']: img for img in coco_data['images']}

    # Группируем предсказания по image_id
    preds_by_image = defaultdict(list)
    for pred in predictions:
        image_id = pred['image_id']
        preds_by_image[image_id].append(pred)

    # Формируем выходные данные в COCO-формате
    output = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "categories": coco_data['categories'],
        "images": [],
        "annotations": []
    }

    # Заполняем информацию об изображениях и аннотациях
    for image_id, preds in preds_by_image.items():
        if image_id not in img_info:
            print(f"Warning: image_id {image_id} not found in original annotations")
            continue

        output['images'].append(img_info[image_id])

        for pred in preds:
            annotation = {
                "id": len(output['annotations']) + 1,  # Уникальный ID аннотации
                "image_id": image_id,
                "category_id": pred['category_id'],
                "bbox": pred['bbox'],  # Должен быть в формате [x,y,width,height]
                "score": pred['score'],  # Добавляем confidence score
                "iscrowd": 0  # По умолчанию не crowd
            }
            output['annotations'].append(annotation)

    # Сохраняем результат
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Successfully converted predictions to COCO format. Saved to {output_file}")



# Пример использования
if __name__ == "__main__":
    # Укажите ваши пути к файлам
    predictions_file = r"X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\submit\skb_test\results.bbox.json"  # Ваш файл с предсказаниями
    annotations_file = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test\skb_test.json'  # COCO-аннотации
    output_file = r"X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\submit\skb_test\coco_predictions.json"  # Выходной файл

    convert_predictions_to_coco(predictions_file, annotations_file, output_file)

