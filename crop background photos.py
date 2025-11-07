import os
import random
from PIL import Image
import argparse
from tqdm import tqdm


def crop_and_save_regions(input_folder, output_folder, n=1, m=5,
                          vertical_range=(0.0, 1.0), target_size=(224, 224)):
    """
    Вырезает области из изображений произвольного разрешения и сохраняет их как отдельные файлы

    Args:
        input_folder (str): Папка с исходными изображениями
        output_folder (str): Папка для сохранения вырезанных областей
        n (int): Из каждого n-го изображения брать области
        m (int): Количество областей для вырезания из каждого изображения
        vertical_range (tuple): Вертикальный диапазон для вырезания (от 0 до 1)
        target_size (tuple): Размер вырезанных областей (ширина, высота)
    """

    # Создаем папку для результатов, если она не существует
    os.makedirs(output_folder, exist_ok=True)

    # Получаем список всех изображений
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(input_folder)
                   if os.path.splitext(f)[1].lower() in image_extensions]

    # Сортируем файлы для воспроизводимости
    image_files.sort()

    # Счетчик для имен файлов
    counter = 0

    # Обрабатываем каждое n-ое изображение
    for i, image_file in tqdm(enumerate(image_files)):
        if i % n != 0:
            continue

        try:
            # Открываем изображение
            image_path = os.path.join(input_folder, image_file)
            with Image.open(image_path) as img:
                # Конвертируем в RGB если нужно
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                img_width, img_height = img.size
                target_width, target_height = target_size

                # Проверяем, что изображение достаточно большое
                if img_width < target_width or img_height < target_height:
                    print(f"Пропускаем {image_file}: слишком маленькое ({img_width}x{img_height})")
                    continue

                # Вычисляем вертикальные границы
                vert_start = int(img_height * vertical_range[0])
                vert_end = int(img_height * vertical_range[1])

                # Проверяем, что вертикальная область достаточно большая
                available_vert_height = vert_end - vert_start
                if available_vert_height < target_height:
                    print(f"Предупреждение: Вертикальная область слишком мала для {image_file}")
                    print(f"Доступно: {available_vert_height}px, нужно: {target_height}px")
                    continue

                # Вычисляем границы третей по горизонтали
                third_width = img_width // 3
                left_third_range = (0, third_width - target_width)
                right_third_range = (2 * third_width, img_width - target_width)

                # Проверяем, что в третях достаточно места для вырезания
                if left_third_range[1] < left_third_range[0]:
                    print(f"Левая треть слишком узкая в {image_file}")
                    left_third_available = False
                else:
                    left_third_available = True

                if right_third_range[1] < right_third_range[0]:
                    print(f"Правая треть слишком узкая в {image_file}")
                    right_third_available = False
                else:
                    right_third_available = True

                if not left_third_available and not right_third_available:
                    print(f"Пропускаем {image_file}: невозможно вырезать области в указанных третях")
                    continue

                # Вырезаем m областей
                regions_cut = 0
                attempts = 0
                max_attempts = m * 10  # Максимальное количество попыток

                while regions_cut < m and attempts < max_attempts:
                    attempts += 1

                    # Выбираем доступную треть
                    available_thirds = []
                    if left_third_available:
                        available_thirds.append('left')
                    if right_third_available:
                        available_thirds.append('right')

                    if not available_thirds:
                        break

                    chosen_third = random.choice(available_thirds)

                    if chosen_third == 'left':
                        x_min = random.randint(left_third_range[0], left_third_range[1])
                    else:  # 'right'
                        x_min = random.randint(right_third_range[0], right_third_range[1])

                    # Случайная позиция по вертикали в заданном диапазоне
                    y_min = random.randint(vert_start, vert_end - target_height)

                    # Вырезаем область
                    crop_box = (x_min, y_min, x_min + target_width, y_min + target_height)
                    cropped_img = img.crop(crop_box)

                    # Сохраняем вырезанную область
                    output_filename = image_file.replace('.png','') + f"_crop_{counter%m:02d}.png"
                    output_path = os.path.join(output_folder, output_filename)
                    cropped_img.save(output_path, 'PNG')

                    counter += 1
                    regions_cut += 1
                    # print(f"Сохранено: {output_filename} из {image_file} (область {regions_cut}/{m})")

                if regions_cut < m:
                    print(f"Вырезано только {regions_cut} из {m} областей для {image_file}")

        except Exception as e:
            print(f"Ошибка при обработке {image_file}: {e}")

    print(f"Готово! Сохранено {counter} вырезанных областей.")


def main():
    parser = argparse.ArgumentParser(description='Вырезание областей из изображений произвольного разрешения')
    parser.add_argument('--input', '-i', required=True, help='Папка с исходными изображениями')
    parser.add_argument('--output', '-o', required=True, help='Папка для сохранения результатов')
    parser.add_argument('--n', type=int, default=1, help='Из каждого n-го изображения (по умолчанию: 1)')
    parser.add_argument('--m', type=int, default=5, help='Количество областей из каждого изображения (по умолчанию: 5)')
    parser.add_argument('--vertical_min', type=float, default=0.0, help='Минимальная вертикальная позиция (0-1)')
    parser.add_argument('--vertical_max', type=float, default=1.0, help='Максимальная вертикальная позиция (0-1)')
    parser.add_argument('--target_width', type=int, default=224, help='Ширина вырезаемой области')
    parser.add_argument('--target_height', type=int, default=224, help='Высота вырезаемой области')

    args = parser.parse_args()

    # Проверяем корректность вертикального диапазона
    if not (0 <= args.vertical_min < args.vertical_max <= 1):
        print("Ошибка: vertical_min должен быть в диапазоне [0,1) и vertical_max в (vertical_min, 1]")
        return

    crop_and_save_regions(
        input_folder=args.input,
        output_folder=args.output,
        n=args.n,
        m=args.m,
        vertical_range=(args.vertical_min, args.vertical_max),
        target_size=(args.target_width, args.target_height)
    )


if __name__ == "__main__":
    # Пример использования напрямую в коде:

    crop_and_save_regions(
        input_folder="Data/background",
        output_folder="Data/background_crop",
        n=6,           # Из каждого n изображения
        m=6,           # m областей из каждого
        vertical_range=(0.0, 0.8),  # Область от 30% до 60% высоты
        target_size=(224, 224)
    )

    # main()