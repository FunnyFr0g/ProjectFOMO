import os
import json
import cv2
import numpy as np
from sympy import false
from torchvision import transforms
import torch
from torchvision.models import mobilenet_v2
import torch.nn as nn
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from clearml import Task, Dataset
import zipfile
from tqdm import tqdm
import matplotlib.pyplot as plt
from FOMOmodels import FomoModelResV0, FomoModelResV1, FomoModel56

from label_pathes import gt_pathes

use_clearml = False

if use_clearml:
    Task.ignore_requirements('pywin32')
    task = Task.init(project_name="SmallObjectDetection",
                     task_name="FOMO 35e res drones_only_FOMO_val",
                     )
    # task.execute_remotely(queue_name='default', exit_process=True)

params = {
"NUM_CLASSES" : 2,  # Кол-во классов (включая фон)
'DO_RESIZE' : True,
"INPUT_SIZE" : (224, 224), # Размер входного изображения
"BOX_SIZE" : 8,  # Размер стороны квадратного желаемого bounding box'а в пикселях
}
if use_clearml:
    params = task.connect(params)

# Константы
TRUNK_AT = 4
NUM_CLASSES = 2

# Чтобы подружить кириллицу с CV2 Сохраняем оригинальные функции
_original_imread = cv2.imread
_original_imwrite = cv2.imwrite

def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """Замена для cv2.imread с поддержкой юникода"""
    try:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8)
                return cv2.imdecode(data, flags)
        return None
    except:
        # Fallback к оригинальной функции
        return _original_imread(path, flags)

def imwrite_unicode(filename, img, params=None):
    """Замена для cv2.imwrite с поддержкой юникода"""
    try:
        ext = os.path.splitext(filename)[1]
        success, encoded_img = cv2.imencode(ext, img, params)
        if success:
            with open(filename, 'wb') as f:
                encoded_img.tofile(f)
            return True
        return False
    except:
        return _original_imwrite(filename, img, params)

# Заменяем функции
cv2.imread = imread_unicode
cv2.imwrite = imwrite_unicode


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


if torch.cuda.is_available():
    MEAN_GPU = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    STD_GPU = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
else:
    MEAN_CPU = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD_CPU = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def prepare_image(image_path, new_size=params["INPUT_SIZE"], to_cuda=True, to_cpu=False):
    """Оптимизированная подготовка изображения (ресайз, и нормализация). Может выполняться на CUDA"""
    # Чтение
    image = cv2.imread(image_path)
    orig_h, orig_w = image.shape[:2]

    # BGR → RGB и ресайз
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if params['DO_RESIZE']:
        image = cv2.resize(image, new_size)

    # Быстрая конвертация в tensor
    # np.ascontiguousarray важно для производительности!
    image = np.ascontiguousarray(image)
    tensor = torch.from_numpy(image).float()

    # Реорганизация и нормализация
    tensor = tensor.permute(2, 0, 1).div_(255.0)  # CHW и [0,1]

    # На GPU (если доступно)
    if to_cuda and torch.cuda.is_available():
        tensor = tensor.cuda()
        tensor = tensor.unsqueeze(0)  # Добавляем batch dimension
        tensor = (tensor - MEAN_GPU) / STD_GPU
    else:
        tensor = tensor.unsqueeze(0)
        tensor = (tensor - MEAN_CPU) / STD_CPU
    if to_cpu:
        tensor = tensor.cpu()

    return tensor, (orig_w, orig_h)


def scale_coords(coords, from_size=(56, 56), to_size=params["INPUT_SIZE"], orig_size=None):
    """Масштабирование координат из feature map в оригинальное разрешение"""
    if orig_size is None:
        orig_size = to_size

    orig_size = orig_size[::-1]  # очень важный фикс для неквадратного изображения (хотя чето нет)
    # Сначала масштабируем к входному размеру модели (224x224)
    y_scale = to_size[0] / from_size[0]
    x_scale = to_size[1] / from_size[1]

    # Затем масштабируем к оригинальному разрешению изображения
    y_scale *= orig_size[0] / to_size[0]
    x_scale *= orig_size[1] / to_size[1]


    # # Сначала масштабируем к входному размеру модели (224x224)
    # x_scale = to_size[0] / from_size[0]
    # y_scale = to_size[1] / from_size[1]
    #
    # # Затем масштабируем к оригинальному разрешению изображения
    # x_scale *= orig_size[0] / to_size[0]
    # y_scale *= orig_size[1] / to_size[1]


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


def main(model=FomoModelResV0(), checkpoint_path=None, model_name='', dataset_path=None, reference_json=None, dataset_name='ds', dir_prefix='', draw_bbox=False, ):
    '''

    Args:
        model: Модель, которую нужно импортировать из FOMOmodels
        checkpoint_path: путь к .pth файлу
        model_name: человеческое имя модели
        dataset_path: папка с изображениями
        reference_json: разметка, с которой нужно синхронизировать порядко изображений
        dataset_name: человеческое имя датасета
        dir_prefix: префикс для итоговой папки. 'predictions/dir_prefix+f'{dataset_name}!{model_name}'
        draw_bbox: Сохранять ли картинки с ббоксами

    Returns:


    '''
    # FomoModel(num_classes=NUM_CLASSES)
    # checkpoint_path = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\FOMO\checkpoints\checkpoint_epoch_100.pth'
    # checkpoint_path = 'weights/BEST_FOMO_56_crossEntropy_dronesOnly_104e_model_weights.pth'
    # checkpoint_path = 'weights/FOMO_56-res-v0_drones_only_FOMO_1.0.2/BEST_35e.pth'
    # checkpoint_path = 'weights/BEST_FOMO_56_crossEntropy_drones_only_FOMO_1.0.1_14e_model_weights.pth'
    # checkpoint_path = 'weights/LATEST.pth'

    if not model_name:
        model_name = checkpoint_path.split('/')[-1].replace('.pth','')

    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # dataset_name = "drones_only"
    # coco_dataset = Dataset.get(dataset_name=dataset_name, dataset_project="SmallObjectDetection")
    # coco_dataset = Dataset.get(dataset_id='ae8c12c33b324947af9ae6379d920eb8') # drones only 1.0.5
    # dataset_path = coco_dataset.get_local_copy()
    img_dir = f"{dataset_path}/val/images"
    if not os.path.exists(img_dir):
        img_dir = f"{dataset_path}/images/val"
    if not os.path.exists(img_dir):
        img_dir = dataset_path # Случай, когда папка с изображениями указывается напрямую
    if not os.path.exists(img_dir):
        raise Exception('Неверный путь к изображениям', img_dir)

    #
    # dataset_name = "drones_only"
    # coco_dataset = Dataset.get(dataset_name=dataset_name, dataset_project="YOLOv5", dataset_version='1.0.5')
    # dataset_path = coco_dataset.get_local_copy()
    # img_dir = f"{dataset_path}/images/val"

    output_dir = os.path.join('predictions', dir_prefix +f'{dataset_name}!{model_name}')
    output_json = os.path.join(output_dir, 'annotations', 'predictions.json')
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    all_annotations = []
    all_images = []
    image_id = 0

    start_time = time.time()

    # Если указан эталонный JSON, загружаем маппинг filename -> image_id
    filename_to_id = {}
    if reference_json and os.path.exists(reference_json):
        with open(reference_json, 'r') as f:
            ref_data = json.load(f)
        filename_to_id = {img['file_name']: img['id'] for img in ref_data['images']}
        print(f"Загружен эталонный JSON с {len(filename_to_id)} изображениями")

    counter = 0

    for img_name in tqdm(os.listdir(img_dir)):
        counter += 1
        # if counter == 100:
        #     break
        if img_name.endswith('.json'):
            continue

        # Определяем image_id в зависимости от наличия эталонного JSON
        if reference_json and img_name in filename_to_id:
            # Используем image_id из эталонного файла
            image_id = filename_to_id[img_name]
        else:
            # Используем автоинкремент, начиная с 1
            image_id += 1
            if reference_json and img_name not in filename_to_id:
                print(f"Предупреждение: {img_name} не найден в эталонном JSON, присвоен ID: {image_id}")
                raise Exception('Нет файла, лучше убрать этот JSON')

        # image_id = int(img_name.strip('.jpg'))
        img_path = os.path.join(img_dir, img_name)
        image_tensor, orig_size = prepare_image(img_path)

        with torch.no_grad():
            output = model(image_tensor.cuda())
            # probs = F.softmax(output, dim=1).squeeze(0).numpy()  # [C, H, W]
            probs = F.softmax(output, dim=1).squeeze(0).cpu().numpy()  # [C, H, W]
            # print(f'{probs.shape=}')
            # pred_mask = torch.argmax(output, dim=1).squeeze().numpy()  # [H, W]
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # [H, W]
            # print(f'{pred_mask.shape=}')


        def show_mask():
            source_img = plt.imread(img_path)
            prep_image = image_tensor[0].permute(1, 2, 0).cpu().numpy()
            heatmap = probs[1]

            fig, axes = plt.subplots(1, 4, figsize=(15, 5))

            axes[0].imshow(source_img)
            axes[0].set_title('Исходное изображение')
            axes[1].imshow(prep_image,)
            axes[1].set_title("Подготовленное изображение")
            axes[2].imshow(heatmap,)
            axes[2].set_title("Heatmap")
            axes[3].imshow(pred_mask, cmap='gray')
            axes[3].set_title("Mask")


            # plt.figure(1)
            # plt.imshow(source_img)
            #

            # plt.figure(2)
            # plt.imshow(heatmap)
            #
            # plt.figure(3)
            # plt.imshow(pred_mask, cmap='gray')

            print(f'{image_id=}')
            print(f'{img_path=}')
            print(pred_mask.shape)
            print(pred_mask)

            plt.show()

        if 1 and pred_mask.any():
            show_mask()



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

    # print(elapsed_time)
    # print(elapsed_time/image_id)

    # Сохраняем результаты
    save_to_coco_format(all_images, all_annotations, output_json)
    # print(f"Predictions saved to {output_json}")

    if use_clearml:
        task.upload_artifact('predictions.json', output_json)

    if draw_bbox:
        zip_folder_preserve_structure(output_dir, output_dir+".zip")
        if use_clearml:
            task.upload_artifact('bbox_images', output_dir+".zip")

    result = f"'{dataset_name} {model_name}': r'{output_json}',"
    print('\n\t', result, '\n')
    return result

class TrainedFomo:
    def __init__(self, model, checkpoint, model_name):
        self.model = model()
        self.checkpoint = checkpoint
        self.model_name = model_name


base = TrainedFomo(
    model=FomoModel56,
    checkpoint=r'weights/BEST_FOMO_56_crossEntropy_dronesOnly_104e_model_weights.pth',
    model_name='FOMO_56_104e'
)

bg = TrainedFomo(
    model=FomoModel56,
    checkpoint='weights/BEST_FOMO_56_crossEntropy_drones_only_FOMO_1.0.1_14e_model_weights.pth',
    model_name='FOMO_bg_56_14e'
)

bg_crop = TrainedFomo(
    model=FomoModel56,
    checkpoint='weights/FOMO_56_bg_crop_drones_only_FOMO_1.0.2/BEST_22e.pth',
    model_name='FOMO_56_22e_bg_crop'
)

res_v0 = TrainedFomo(
    model=FomoModelResV0,
    checkpoint='weights/FOMO_56-res-v0_drones_only_FOMO_1.0.2/BEST_35e.pth',
    model_name='FOMO_56_35e_res_v0'
)

res_v0_focal = TrainedFomo(
    model=FomoModelResV0,
    checkpoint='weights/FOMO_56_res_v0_focal drones_only_FOMO_1.0.2/BEST_49e.pth',
    model_name='FOMO_56_42e_res_v0_focal'
)

res_v0_focal_896 = TrainedFomo(
    model=FomoModelResV0,
    checkpoint='weights/FOMO_56_res_v0_focal drones_only_FOMO_1.0.2/BEST_49e.pth',
    model_name='FOMO_56_42e_res_v0_focal_896p'
)



res_v1 = TrainedFomo(
    model=FomoModelResV1,
    checkpoint='weights/FOMO_56_res_v1 drones_only_FOMO_1.0.2/BEST_42e.pth',
    model_name='FOMO_56_42e_res_v1'
)

res_v1_896 = TrainedFomo(
    model=FomoModelResV1,
    checkpoint='weights/FOMO_56_res_v1 drones_only_FOMO_1.0.2/BEST_42e.pth',
    model_name='FOMO_56_42e_res_v1_896p'
)

res_v1_focal = TrainedFomo(
    model=FomoModelResV1,
    checkpoint='weights/FOMO_56_res_v1_focal drones_only_FOMO_1.0.2/BEST_71e.pth',
    model_name='FOMO_56_71e_res_v1_focal'
)

class MyDataset:
    def __init__(self, name, image_path, reference_json):
        self.name = name
        self.image_path = image_path
        self.reference_json = reference_json

drones_only_FOMO_val = MyDataset(
    image_path=Dataset.get(dataset_name='drones_only_FOMO', dataset_project="SmallObjectDetection").get_local_copy(),
    reference_json = gt_pathes['drones_only_FOMO_val'],
    name='drones_only_FOMO_val'
)

drones_only_val = MyDataset(
    image_path=Dataset.get(dataset_id='ae8c12c33b324947af9ae6379d920eb8').get_local_copy(),  # drones only 1.0.5
    reference_json = 'GTlabels/ae8c12c33b324947af9ae6379d920eb8/coco_annotations_val.json',
    name='drones_only_val',
)

vid1_drone = MyDataset(
    image_path = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\vid1\images', #vid 1 drone
    reference_json = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\vid1\labels_drone\annotations\drone_annotations.json',
    name='vid1_drone',
)


skb_test = MyDataset(
    name='skb_test',
    image_path=r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test\images', # skb_test
    reference_json = r'GTlabels/skb_test/skb_test.json'
)

skb_test_nobird = MyDataset(
    image_path=r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test_nobird\images',  # skb_test_nobird
    reference_json = r'GTlabels/skb_test_nobird/annotations.json',
    name='skb_test_nobird'
)

skb_test_bg = MyDataset(
    image_path=r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test_bg\images',  #
    reference_json = r'GTlabels/skb_test_bg/annotations.json',
    name='skb_test_bg'
)

synth_drone_val = MyDataset(
    image_path=Dataset.get(dataset_id='f9ad9ea5bc3d40498e67133578ad5099').get_local_copy(),  # synth_drone_val
    reference_json = gt_pathes['synth_drone_val'],
    name='synth_drone_val'
)




if __name__ == '__main__':

    all_models = [base, bg, bg_crop, res_v0, res_v0_focal, res_v1, res_v1_focal]
    models = [res_v1_896]
    datasets = [drones_only_FOMO_val, drones_only_val, vid1_drone,skb_test, skb_test_nobird, skb_test_bg, synth_drone_val]
    # datasets = [synth_drone_val]

    result_pathes = []

    for dataset in datasets:
        for trained_model in models:
            result = main(model=trained_model.model,
                 checkpoint_path=trained_model.checkpoint,
                 model_name=trained_model.model_name,
                 dataset_path=dataset.image_path,
                 reference_json=dataset.reference_json,
                 dataset_name=dataset.name,

                 draw_bbox=False)
            result_pathes.append(result)

    print('Done!')
    for result_path in result_pathes:
        print(result_path)
    input("Press Enter to continue...")
    exit(0)

    # reference_json = None
    model = FomoModelResV0()
    # checkpoint_path = r'weights/FOMO_56_res_v1_focal drones_only_FOMO_1.0.2/BEST_71e.pth'
    # checkpoint_path = r'weights/FOMO_56_res_v1_focal drones_only_FOMO_1.0.2/BEST_42e (1).pth'
    checkpoint_path = r'weights/FOMO_56_res_v0_focal drones_only_FOMO_1.0.2/BEST_49e.pth'
    model_name = 'FOMO_56_42e_res_v0_focal'

    # model = FomoModel56() # Дефолтная FOMO
    # checkpoint_path = r'weights/BEST_FOMO_56_crossEntropy_dronesOnly_104e_model_weights.pth'
    # model_name = 'FOMO_56_104e'


    # coco_dataset = Dataset.get(dataset_name='drones_only_FOMO', dataset_project="SmallObjectDetection")
    # image_path = coco_dataset.get_local_copy()
    # reference_json = gt_pathes['drones_only_FOMO_val']
    # dataset_name = 'drones_only_FOMO_val'

    # main(model=model, checkpoint_path=checkpoint_path, model_name=model_name,
    #      dataset_path=image_path, reference_json=reference_json, dataset_name=dataset_name, draw_bbox=False)

    # coco_dataset = Dataset.get(dataset_id='ae8c12c33b324947af9ae6379d920eb8')  # drones only 1.0.5
    # image_path = coco_dataset.get_local_copy()
    # reference_json = 'GTlabels/ae8c12c33b324947af9ae6379d920eb8/coco_annotations_val.json'
    # dataset_name = 'drones_only_val'

    # image_path = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\vid1\images' #vid 1 drone
    # reference_json = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\vid1\labels_drone\annotations\drone_annotations.json'
    # dataset_name = 'vid1_drone'

    # image_path = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test\images' # skb_test
    # reference_json = r'GTlabels/skb_test/skb_test.json'
    # dataset_name = 'skb_test'

    image_path = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test_nobird\images' # skb_test_nobird
    reference_json = r'GTlabels/skb_test_nobird/annotations.json'
    dataset_name = 'skb_test_nobird'

    main(model=model, checkpoint_path=checkpoint_path, model_name=model_name,
         dataset_path=image_path, reference_json=reference_json, dataset_name=dataset_name, draw_bbox=False)





