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
BOX_SIZE = 8  # Размер стороны квадратного bounding box'а в пикселях
TRUNK_AT = 4
NUM_CLASSES = 2


class FomoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = mobilenet_v2(pretrained=False).features[:TRUNK_AT]

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




def process_predictions(pred_mask, pred_probs, orig_size, image_id):
    """Обработка предсказаний с добавлением confidence score и объединением близких пикселей"""
    annotations = []
    annotation_id = 1

    # Конвертируем маску в uint8 и одноканальное изображение
    kernel = np.ones((3, 3), np.uint8)

    for class_id in range(1, NUM_CLASSES):
        class_mask = (pred_mask == class_id).astype(np.uint8)

        # Убедимся, что маска одноканальная и в правильном формате
        if len(class_mask.shape) > 2:
            class_mask = class_mask.squeeze()

        # # Морфологические операции для удаления шума
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
            half_size = BOX_SIZE // 2
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
from FOMOmodels import FomoModel112



model = FomoModel112() # FomoModel(num_classes=NUM_CLASSES)
# checkpoint_path = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\FOMO\checkpoints\checkpoint_epoch_100.pth'
checkpoint_path = 'FOMO_112_focalloss_10e_model_weights.pth'
state_dict = torch.load(checkpoint_path)
# print(state_dict)
model.load_state_dict(state_dict)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Всего параметров в модели: {total_params}")

# img_dir = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train_FOMO\val\images'
# img_dir = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\vid2\screens'
# img_dir = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\vid1\images'
img_dir = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test\images'
output_json = os.path.join(img_dir, 'FOMO_112p_10e_predictions.json')

all_annotations = []
all_images = []
image_id = 0
import time

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

end_time = time.time()

elapsed_time = end_time - start_time

print(elapsed_time)
print(elapsed_time/image_id)

# Сохраняем результаты
save_to_coco_format(all_images, all_annotations, output_json)
print(f"Predictions saved to {output_json}")