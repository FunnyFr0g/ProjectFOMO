import os
import shutil
from pycocotools.coco import COCO

# Создаем структуру, которую ожидает YOLO
dataset_dir = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train'
os.makedirs(f'{dataset_dir}/train/images', exist_ok=True)
os.makedirs(f'{dataset_dir}/train/labels', exist_ok=True)
os.makedirs(f'{dataset_dir}/val/images', exist_ok=True)
os.makedirs(f'{dataset_dir}/val/labels', exist_ok=True)

# Загружаем аннотации COCO
train_ann_file = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\annotations\split_train_coco.json'

val_ann_file = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\annotations\split_val_coco.json'

coco_train = COCO(train_ann_file)
coco_val = COCO(val_ann_file)

# Функция для конвертации COCO в YOLO формат
def convert_coco_to_yolo(coco, img_dir, output_img_dir, output_label_dir):
    img_ids = coco.getImgIds()
    for c, img_id in enumerate(img_ids):
        if ~c//10000:
          print(f'{c/len(img_ids)*100 :.2f}%')

        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])

        # Копируем изображение
        shutil.copy(img_path, output_img_dir)

        # Получаем аннотации
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Создаем файл с разметкой YOLO
        label_path = os.path.join(output_label_dir, os.path.splitext(img_info['file_name'])[0] + '.txt')
        with open(label_path, 'w') as f:
            for ann in anns:
                # Конвертируем COCO bbox в YOLO формат
                x, y, w, h = ann['bbox']
                img_w, img_h = img_info['width'], img_info['height']
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h

                # Записываем в файл: class_id x_center y_center width height
                f.write(f"{ann['category_id']} {x_center} {y_center} {w_norm} {h_norm}\n")

# Конвертируем тренировочные и валидационные данные
convert_coco_to_yolo(
    coco_train,
    r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\images',
    f'{dataset_dir}/train/images',
    f'{dataset_dir}/train/labels'
)

convert_coco_to_yolo(
    coco_val,
    r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\images',
    f'{dataset_dir}/val/images',
    f'{dataset_dir}/val/labels'
)