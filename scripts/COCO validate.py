import json
from jsonschema import validate
from jsonschema.exceptions import ValidationError

# Схема COCO формата (упрощенная версия)
COCO_SCHEMA = {
    "type": "object",
    "required": ["info", "licenses", "images", "annotations", "categories"],
    "properties": {
        "info": {
            "type": "object",
            "required": ["description", "url", "version", "year", "contributor", "date_created"],
            "properties": {
                "description": {"type": "string"},
                "url": {"type": "string"},
                "version": {"type": "string"},
                "year": {"type": "integer"},
                "contributor": {"type": "string"},
                "date_created": {"type": "string"}
            }
        },
        "licenses": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["url", "id", "name"],
                "properties": {
                    "url": {"type": "string"},
                    "id": {"type": "integer"},
                    "name": {"type": "string"}
                }
            }
        },
        "images": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "width", "height", "file_name"],
                "properties": {
                    "id": {"type": "integer"},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                    "file_name": {"type": "string"},
                    "license": {"type": "integer"},
                    "flickr_url": {"type": "string"},
                    "coco_url": {"type": "string"},
                    "date_captured": {"type": "string"}
                }
            }
        },
        "annotations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "image_id", "category_id", "segmentation", "area", "bbox", "iscrowd"],
                "properties": {
                    "id": {"type": "integer"},
                    "image_id": {"type": "integer"},
                    "category_id": {"type": "integer"},
                    "segmentation": {
                        "oneOf": [
                            {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                            {"type": "object"}  # RLE формат
                        ]
                    },
                    "area": {"type": "number"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4
                    },
                    "iscrowd": {"type": "integer", "enum": [0, 1]}
                }
            }
        },
        "categories": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name", "supercategory"],
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "supercategory": {"type": "string"}
                }
            }
        }
    }
}


def validate_coco(file_path):
    """Проверяет, соответствует ли файл формату COCO"""
    try:
        with open(file_path, 'r') as f:
            coco_data = json.load(f)

        validate(instance=coco_data, schema=COCO_SCHEMA)
        print("✅ Файл соответствует формату COCO")
        return True
    except ValidationError as e:
        print(f"❌ Ошибка валидации: {e.message}")
        print(f"Путь к ошибке: {' -> '.join(map(str, e.absolute_path))}")
        return False
    except json.JSONDecodeError:
        print("❌ Ошибка: Файл не является валидным JSON")
        return False
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {str(e)}")
        return False


def check_data_consistency(coco_data):
    """Проверяет согласованность данных (ссылочная целостность)"""
    errors = []

    # Собираем все ID для проверки ссылок
    image_ids = {img['id'] for img in coco_data['images']}
    category_ids = {cat['id'] for cat in coco_data['categories']}
    annotation_ids = set()

    # Проверяем аннотации
    for ann in coco_data['annotations']:
        if ann['id'] in annotation_ids:
            errors.append(f"Дублирующийся ID аннотации: {ann['id']}")
        annotation_ids.add(ann['id'])

        if ann['image_id'] not in image_ids:
            errors.append(f"Аннотация {ann['id']} ссылается на несуществующее изображение {ann['image_id']}")

        if ann['category_id'] not in category_ids:
            errors.append(f"Аннотация {ann['id']} ссылается на несуществующую категорию {ann['category_id']}")

    if errors:
        print("❌ Найдены проблемы согласованности данных:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ Данные согласованы (ссылочная целостность в порядке)")
        return True


def full_coco_validation(file_path):
    """Полная проверка COCO файла (схема + согласованность данных)"""
    try:
        with open(file_path, 'r') as f:
            coco_data = json.load(f)

        # Проверка схемы
        validate(instance=coco_data, schema=COCO_SCHEMA)
        print("✅ Файл соответствует формату COCO (схема)")

        # Проверка согласованности данных
        return check_data_consistency(coco_data)

    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        return False


# Пример использования
if __name__ == "__main__":
    import sys

    # if len(sys.argv) != 2:
    #     print("Использование: python validate_coco.py <путь_к_json_файлу>")
    #     sys.exit(1)

    file_path = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\val\mva23_val.json'
    file_path = r'FOMO_50e_predictions.json'
    print(f"Проверка файла: {file_path}")

    # Простая проверка схемы
    validate_coco(file_path)

    # Или полная проверка (схема + согласованность данных)
    full_coco_validation(file_path)