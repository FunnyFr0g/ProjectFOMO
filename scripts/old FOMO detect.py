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
from skimage.measure import regionprops

# Константы
BOX_SIZE = 6  # Размер стороны квадратного bounding box'а в пикселях
TRUNK_AT = 4
NUM_CLASSES = 2


class FomoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = mobilenet_v2(pretrained=True).features[:TRUNK_AT]

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
    image = cv2.imread(image_path)
    orig_h, orig_w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(image)
    return image_tensor.unsqueeze(0), (orig_w, orig_h)


def scale_coords(coords, from_size=(56, 56), to_size=(224, 224), orig_size=None):
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


def process_predictions(pred_mask, orig_size, image_id):
    """Обработка предсказаний и генерация COCO-аннотаций"""
    annotations = []
    annotation_id = 1  # Счетчик для уникальных ID аннотаций

    # Находим связанные компоненты для каждого класса
    for class_id in range(1, NUM_CLASSES):
        class_mask = (pred_mask == class_id).astype(np.uint8)

        # Находим центроиды объектов
        regions = regionprops(class_mask)
        print(len(regions))
        print(regions)

        for region in regions:
            # Координаты центроида в feature map (56x56)
            y_centroid, x_centroid = region.centroid

            # Масштабируем координаты к оригинальному разрешению
            scaled_coords = scale_coords([(y_centroid, x_centroid)],
                                         from_size=pred_mask.shape,
                                         to_size=(224, 224),
                                         orig_size=orig_size)

            y_orig, x_orig = scaled_coords[0]

            # Создаем bounding box (6x6 пикселей)
            half_size = BOX_SIZE // 2
            x1 = max(0, x_orig - half_size)
            y1 = max(0, y_orig - half_size)
            x2 = min(orig_size[0], x_orig + half_size)
            y2 = min(orig_size[1], y_orig + half_size)

            # Формируем аннотацию в COCO-формате
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
                "centroid": [x_orig, y_orig]  # Добавляем центроид
            }

            print(annotation)

            annotations.append(annotation)
            annotation_id += 1

    return annotations


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
model = FomoModel(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load('FOMO_56_focalloss_50e_model_weights.pth'))
model.eval()

img_dir = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train_FOMO\val\images'
output_json = 'old_FOMO_predictions.json'

all_annotations = []
all_images = []
image_id = 1

for img_name in os.listdir(img_dir):
    image_id = int(img_name.strip('.jpg'))
    img_path = os.path.join(img_dir, img_name)
    image_tensor, orig_size = prepare_image(img_path)

    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        pred_mask = probs.squeeze().cpu().numpy()
        print(pred_mask)
        # pred_mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy()

    # Создаем запись об изображении
    image_info = {
        "id": image_id,
        "file_name": img_name,
        "width": orig_size[0],
        "height": orig_size[1]
    }
    all_images.append(image_info)

    # Обрабатываем предсказания
    annotations = process_predictions(pred_mask, orig_size, image_id)
    all_annotations.extend(annotations)

    # image_id += 1

# Сохраняем результаты
save_to_coco_format(all_images, all_annotations, output_json)
print(f"Predictions saved to {output_json}")