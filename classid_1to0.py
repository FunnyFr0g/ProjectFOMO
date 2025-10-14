import json


def update_category_ids(input_file1, input_file2, output_file1, output_file2, new_class_name):
    """
    Заменяет category_id с 1 на 0 и меняет название класса в COCO JSON файлах

    Args:
        input_file1: путь к первому входному файлу
        input_file2: путь ко второму входному файлу
        output_file1: путь к первому выходному файлу
        output_file2: путь ко второму выходному файлу
        new_class_name: новое название класса
    """

    def process_file(input_file, output_file):
        # Читаем JSON файл
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Обновляем category_id в annotations
        annotations_updated = 0
        for annotation in data.get('annotations', []):
            if annotation.get('category_id') == 1:
                annotation['category_id'] = 0
                annotations_updated += 1

        # Обновляем id и name в categories
        categories_updated = 0
        for category in data.get('categories', []):
            if category.get('id') == 1:
                category['id'] = 0
            category['name'] = new_class_name
            categories_updated += 1

        # Сохраняем обновленный файл
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Файл обработан: {input_file} -> {output_file}")
        print(f"  Обновлено аннотаций: {annotations_updated}")
        print(f"  Обновлено категорий: {categories_updated}")
        print(f"  Новое название класса: '{new_class_name}'")

    # Обрабатываем оба файла
    process_file(input_file1, output_file1)
    process_file(input_file2, output_file2)


# Пример использования
if __name__ == "__main__":
    # Укажите пути к вашим файлам
    input_file1 = r"C:\Users\ILYA\PycharmProjects\PythonProject\processed_dataset\train\train_annotations.json"
    input_file2 = r"C:\Users\ILYA\PycharmProjects\PythonProject\processed_dataset\train\train_annotations.json"
    output_file1 = r"C:\Users\ILYA\PycharmProjects\PythonProject\processed_dataset\train\train_annotations_0.json"
    output_file2 = r"C:\Users\ILYA\PycharmProjects\PythonProject\processed_dataset\train\train_annotations_0.json"

    new_class_name = "bird"

    update_category_ids(input_file1, input_file2, output_file1, output_file2, new_class_name)
    print("Оба файла успешно обработаны!")