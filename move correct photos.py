import cv2
import os
import glob
import csv
import numpy as np  # Импортируем numpy
import re


def natural_sort_key(s, natsort_regex=re.compile('([0-9]+)')):
    """
    Функция для "естественной" сортировки строк.
    Разбивает строку на части, состоящие из текста и чисел,
    и преобразует числовые части в целые числа для правильного сравнения.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(natsort_regex, s)]

def image_viewer(image_dir, confirmed_img_dir, confirmed_csv):
    """
    Просмотрщик изображений с перемещением и записью.

    Args:
        image_dir: Путь к папке с изображениями.
        confirmed_img_dir: Путь к папке для перемещенных изображений.
        confirmed_csv: Путь к файлу CSV для записи имен перемещенных файлов.
    """

    image_files = sorted(glob.glob(os.path.join(image_dir, "*.*")))  # Получаем список файлов и сортируем их
    image_files = [f for f in image_files if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.gif',
                                                                                '.bmp']]  # Фильтруем список, оставляя только файлы с расширениями изображений
    image_files = sorted(image_files, key=natural_sort_key)
    if not image_files:
        print(f"В папке {image_dir} не найдено изображений.")
        return

    current_index = 0
    num_images = len(image_files)

    # Создаем папку для перемещенных изображений, если её нет
    if not os.path.exists(confirmed_img_dir):
        os.makedirs(confirmed_img_dir)

    def show_image():
        """Отображает текущее изображение."""
        nonlocal current_index
        image_path = image_files[current_index]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Ошибка: Не удалось прочитать изображение {image_path}")
            # Удаляем некорректный файл из списка, чтобы не зациклиться на нем
            del image_files[current_index]
            num_images -= 1
            if num_images == 0:
                print("Больше нет доступных изображений.")
                cv2.destroyAllWindows()
                return
            if current_index >= num_images:
                current_index = num_images - 1
            show_image()  # Показываем следующее доступное изображение
            return

        print(f"Отображается: {os.path.basename(image_path)}")  # Вывод имени файла
        cv2.namedWindow("Image Viewer", cv2.WINDOW_NORMAL)
        cv2.putText(image, str(os.path.basename(image_path)), (0,50), 0,1, cv2.FONT_ITALIC, 2)
        cv2.imshow("Image Viewer", image)
        return True

    def move_and_record(image_path):
        """Перемещает изображение и записывает имя в CSV."""
        nonlocal current_index
        filename = os.path.basename(image_path)
        new_path = os.path.join(confirmed_img_dir, filename)

        try:
            os.rename(image_path, new_path)  # Перемещаем файл
            print(f"Фото {filename} перемещено")

            with open(confirmed_csv, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([filename])

            # Удаляем перемещенный файл из списка
            del image_files[current_index]
            nonlocal num_images
            num_images -= 1

        except Exception as e:
            print(f"Ошибка при перемещении {filename}: {e}")
            return False
        return True

    if not show_image():
        return

    while True:
        key = cv2.waitKey(0)  # Ожидаем нажатие клавиши

        if key == 27 or key == -1:  # ESC
            break
        elif key == 97 or key == 244:  # кнопка A
            current_index = (current_index - 1) % num_images
            if not show_image():
                break
        elif key == 226 or key == 100:  # кнопка D
            current_index = (current_index + 1) % num_images
            if not show_image():
                break
        elif key == 32:  # Пробел
            image_path = image_files[current_index]
            if move_and_record(image_path):
                # Если успешно перемещено, показываем следующее изображение
                if num_images == 0:
                    print("Больше нет изображений для обработки.")
                    break  # Выходим из цикла, если все изображения обработаны

                if current_index >= num_images:
                    current_index = 0 if num_images > 0 else -1

                if current_index >= 0 and num_images > 0:
                    if not show_image():
                        break
                else:
                    print("Больше нет изображений для обработки.")
                    break
            else:
                # Если перемещение не удалось, остаемся на текущем изображении.
                pass  # Не нужно ничего делать, пользователь может попробовать снова или пропустить изображение
        else:
            print(f"Нажата клавиша: {key}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_dir = r"C:\Users\Ilya\PycharmProjects\test yolo\april spores\datasets\Dataset_3b\bbox images"  # путь к папке с изображениями
    confirmed_img_dir = r"C:\Users\Ilya\PycharmProjects\test yolo\april spores\datasets\Dataset_3b\good bbox images"  # Папка для перемещенных
    confirmed_csv = "confirmed.csv"  # Файл для записи имен перемещенных

    # Создаем папку с изображениями для примера
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        # Создадим несколько тестовых изображений (черные квадраты)
        for i in range(3):
            img = (0 * np.ones((100, 100, 3))).astype('uint8')
            cv2.imwrite(os.path.join(image_dir, f"test_image_{i}.png"), img)

    image_viewer(image_dir, confirmed_img_dir, confirmed_csv)