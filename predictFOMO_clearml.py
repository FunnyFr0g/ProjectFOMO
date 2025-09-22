import os
import json
import cv2
import numpy as np
from torchvision import transforms
import torch
from torchvision.models import mobilenet_v2
import torch.nn as nn
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from clearml import Task, Dataset
import zipfile


use_clearml = False

if use_clearml:
    task = Task.init(project_name="SmallObjectDetection", task_name="PredictFOMO")

    task.execute_remotely(queue_name='default', exit_process=True)

params = {
"NUM_CLASSES" : 2,  # Кол-во классов (включая фон)
"INPUT_SIZE" : (224, 224), # Размер входного изображения
"BOX_SIZE" : 8,  # Размер стороны квадратного желаемого bounding box'а в пикселях
}
if use_clearml:
    params = task.connect(params)

# Константы
TRUNK_AT = 4
NUM_CLASSES = 2


class FomoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = mobilenet_v2(pretrained=False).features[:params['TRUNK_AT']]

    def forward(self, x):
        return self.mobilenet(x)


class FomoHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(24, 48, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(48, 32, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        return x


class FomoModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = FomoBackbone()
        self.head = FomoHead(num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def prepare_image(image_path):
    """Подготовка изображения (ресайз и нормализация)"""
    image = cv2.imread(image_path)
    orig_h, orig_w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(params["INPUT_SIZE"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(image)
    return image_tensor.unsqueeze(0), (orig_w, orig_h)


def scale_coords(coords, from_size=(56, 56), to_size=params["INPUT_SIZE"], orig_size=None):
    """Масштабирование координат из feature map в оригинальное разрешение"""
    if orig_size is None:
        orig_size = to_size

    # Сначала масштабируем к входному размеру модели (224x224)
    x_scale = to_size[0] / from_size[0]
    y_scale = to_size[1] / from_size[1]

    # Затем масштабируем к оригинальному разрешению изображения
    x_scale *= orig_size[0] / to_size[0]
    y_scale *= orig_size[1] / to_size[1]

    scaled_coords = []
    for y, x in coords:
        scaled_coords.append((int(y * y_scale), int(x * x_scale)))
    return scaled_coords




def process_predictions(pred_mask, pred_probs, orig_size, image_id):
    """Обработка предсказаний с добавлением confidence score и объединением близких пикселей"""
    annotations = []
    annotation_id = 1

    # Конвертируем маску в uint8 и одноканальное изображение
    kernel = np.ones((3, 3), np.uint8)

    for class_id in range(1, params['NUM_CLASSES']):
        class_mask = (pred_mask == class_id).astype(np.uint8)

        # Убедимся, что маска одноканальная и в правильном формате
        if len(class_mask.shape) > 2:
            class_mask = class_mask.squeeze()

        # # Морфологические операции для удаления шума из маски (оказалось, делают хуже)
        # cleaned_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
        # cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = class_mask

        # Находим все ненулевые пиксели для текущего класса
        y_coords, x_coords = np.where(cleaned_mask == 1)

        if len(y_coords) == 0:
            continue

        # Создаем список точек и соответствующих scores
        points = list(zip(y_coords, x_coords))
        scores = pred_probs[class_id][y_coords, x_coords]

        # Группируем близлежащие пиксели (в радиусе 4x4)
        processed = np.zeros(len(points), dtype=bool)
        groups = []

        for i in range(len(points)):
            if not processed[i]:
                # Находим все точки в радиусе 4 пикселя
                current_group = [i]
                processed[i] = True

                # Проверяем соседей
                queue = [i]
                while queue:
                    idx = queue.pop(0)
                    y1, x1 = points[idx]

                    for j in range(len(points)):
                        if not processed[j]:
                            y2, x2 = points[j]
                            if abs(y1 - y2) <= 2 and abs(x1 - x2) <= 2:  # 4x4 область (2 пикселя в каждую сторону)
                                processed[j] = True
                                current_group.append(j)
                                queue.append(j)

                groups.append(current_group)

        # Обрабатываем каждую группу
        for group in groups:
            if not group:
                continue

            # Координаты точек в группе
            group_points = [points[i] for i in group]
            group_scores = [scores[i] for i in group]

            # Вычисляем средний центроид и score
            y_coords = [p[0] for p in group_points]
            x_coords = [p[1] for p in group_points]

            y_centroid = np.mean(y_coords)
            x_centroid = np.mean(x_coords)
            score = np.mean(group_scores)

            # Масштабируем координаты
            scaled_coords = scale_coords([(y_centroid, x_centroid)],
                                         from_size=pred_mask.shape,
                                         to_size=(224, 224),
                                         orig_size=orig_size)
            y_orig, x_orig = scaled_coords[0]

            # Bounding box
            half_size = params['BOX_SIZE'] // 2
            x1 = max(0, x_orig - half_size)
            y1 = max(0, y_orig - half_size)
            x2 = min(orig_size[0], x_orig + half_size)
            y2 = min(orig_size[1], y_orig + half_size)

            # Площадь (примерная)
            area = len(group) * (orig_size[0] / pred_mask.shape[1]) * (orig_size[1] / pred_mask.shape[0])

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": area,
                "score": float(score),  # Добавляем confidence score
                "iscrowd": 0,
                "centroid": [x_orig, y_orig]
            }

            annotations.append(annotation)
            annotation_id += 1

    return annotations


def save_image_with_bboxes(image, annotations, output_path, categories=(0,1), confidence_threshold=0):
    """
    Сохраняет изображение с нарисованными рамками и подписями для текущего кадра
    """

    # Создаем копию изображения чтобы не портить оригинал
    image_copy = image.copy()

    # Рисуем bounding boxes для каждой аннотации
    for ann in annotations:
        if 'score' in ann and ann['score'] < confidence_threshold:
            continue

        # Получаем координаты bbox [x, y, width, height]
        bbox = ann['bbox']
        x, y, w, h = map(int, bbox)

        category_id = ann['category_id']
        if categories and category_id in categories:
            category_name = categories[category_id]
        else:
            category_name = f"Class {category_id}"

        # Генерируем текст подписи
        if 'score' in ann:
            label_text = f"{category_name}: {ann['score']:.2f}"
        else:
            label_text = category_name

        # Выбираем цвет на основе категории
        color = get_color_for_category(category_id)

        # Рисуем прямоугольник
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, 2)

        # Получаем размер текста
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Рисуем фон для текста
        cv2.rectangle(image_copy,
                      (x, y - text_height - 5),
                      (x + text_width, y),
                      color, -1)  # -1 означает заливку

        # Рисуем текст
        cv2.putText(image_copy, label_text,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1)  # Белый текст

    # Сохраняем результат
    cv2.imwrite(output_path, image_copy)
    # print(f"Сохранено: {output_path}")


def zip_folder_preserve_structure(folder_path, output_zip_path):
    """
    Архивирует папку, сохраняя полную структуру директорий
    """
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Сохраняем полный путь внутри архива
                arcname = os.path.join(
                    os.path.basename(folder_path),
                    os.path.relpath(file_path, folder_path)
                )
                zipf.write(file_path, arcname)


def get_color_for_category(category_id):
    """Генерирует цвет на основе ID категории"""
    colors = [
        (0, 0, 255),  # Красный (BGR)
        (0, 255, 0),  # Зеленый
        (255, 0, 0),  # Синий
        (0, 255, 255),  # Желтый
        (255, 0, 255),  # Пурпурный
        (255, 255, 0),  # Голубой
        (0, 165, 255),  # Оранжевый
        (128, 0, 128),  # Фиолетовый
        (255, 192, 203),  # Розовый
        (0, 255, 127),  # Весенний зеленый
    ]
    return colors[category_id % len(colors)]


def save_to_coco_format(images, annotations, output_path):
    """Сохранение результатов в COCO-формате"""
    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "bird"}],
        "info": {"description": "FOMO model predictions"},
        "licenses": [{"id": 1, "name": "CC-BY"}]
    }

    with open(output_path, 'w') as f:
        json.dump(coco_output, f, indent=2)


# Основной код
from FOMOmodels import FomoModel56, FomoModel112
import time


def main(model = FomoModel56(), draw_bbox=True):
    # FomoModel(num_classes=NUM_CLASSES)
    # checkpoint_path = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\FOMO\checkpoints\checkpoint_epoch_100.pth'
    checkpoint_path = 'weights/FOMO_56_focalloss_10e_model_weights.pth'

    model_name = checkpoint_path.split('/')[-1].strip('.pth')

    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()

    dataset_name = "FOMO-mva23"
    coco_dataset = Dataset.get(dataset_name=dataset_name, dataset_project="SmallObjectDetection")
    dataset_path = coco_dataset.get_local_copy()

    img_dir = f"{dataset_path}/val/images"

    output_dir = os.path.join('predictions', f'{dataset_name}_{model_name}')
    output_json = os.path.join(output_dir,'annotations', model_name+'_predictions.json')
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    all_annotations = []
    all_images = []
    image_id = 0

    start_time = time.time()

    images_count = len(os.listdir(img_dir))

    for img_name in os.listdir(img_dir):
        if img_name.endswith('.json'):
            continue

        image_id += 1

        if image_id % 100 == 0:
            print(f'{image_id}/{images_count} {image_id/images_count*100 :.2f}%')
        # image_id = int(img_name.strip('.jpg'))
        img_path = os.path.join(img_dir, img_name)
        image_tensor, orig_size = prepare_image(img_path)

        with torch.no_grad():
            output = model(image_tensor)
            # print(f'{output.shape=}')
            probs = F.softmax(output, dim=1).squeeze(0).cpu().numpy()  # [C, H, W]
            # print(f'{probs.shape=}')
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # [H, W]
            # print(f'{pred_mask.shape=}')


            import matplotlib.pyplot as plt
            # if image_id == 34:
            #     print(pred_mask.shape)
            #     print(pred_mask)
            #     plt.imshow(pred_mask)
            #     plt.show()

        # Создаем запись об изображении
        image_info = {
            "id": image_id,
            "file_name": img_name,
            "width": orig_size[0],
            "height": orig_size[1]
        }
        all_images.append(image_info)

        # Обрабатываем предсказания
        annotations = process_predictions(pred_mask, probs, orig_size, image_id)
        all_annotations.extend(annotations)

        # Рисуем bbox
        if draw_bbox:
            image = cv2.imread(img_path)
            output_path = os.path.join(output_dir, img_name)
            save_image_with_bboxes(image, annotations, output_path, categories=(0,1), confidence_threshold=0)

    end_time = time.time()

    elapsed_time = end_time - start_time

    print(elapsed_time)
    print(elapsed_time/image_id)

    # Сохраняем результаты
    save_to_coco_format(all_images, all_annotations, output_json)
    print(f"Predictions saved to {output_json}")

    if use_clearml:
        task.upload_artifact('predictions.json', output_json)

    if draw_bbox:
        zip_folder_preserve_structure(output_dir, output_dir+".zip")
        if use_clearml:
            task.upload_artifact('bbox_images', output_dir+".zip")


if __name__ == '__main__':
    main()

