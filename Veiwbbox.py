import os
import json
import cv2
import numpy as np

from label_pathes import pred_pathes, gt_pathes


class CocoViewer:
    def __init__(self, image_folder, coco_json_path):
        self.image_folder = image_folder
        self.coco_json_path = coco_json_path
        self.image_files = sorted(
            [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.current_index = 0
        self.conf_threshold = 0.5  # Начальный порог уверенности
        self.window_name = "COCO Viewer - A/D: navigate, W/S: conf threshold, ESC: exit"

        # Load COCO data
        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)

        # Create image_id to annotations mapping
        self.image_id_to_annotations = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)

        # Create filename to image_info mapping
        self.filename_to_image_info = {}
        for img_info in self.coco_data['images']:
            self.filename_to_image_info[img_info['file_name']] = img_info

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)

    def get_current_image_info(self):
        if not self.image_files:
            return None
        current_file = self.image_files[self.current_index]
        return self.filename_to_image_info.get(current_file, None)

    def draw_annotations(self, image, image_info):
        if not image_info:
            return image

        image_id = image_info['id']
        annotations = self.image_id_to_annotations.get(image_id, [])

        for ann in annotations:
            # Проверяем score (если есть) и порог
            if 'score' in ann and ann['score'] < self.conf_threshold:
                continue

            # Draw bounding box
            bbox = ann['bbox']
            x, y, w, h = map(int, bbox)
            color = (0, 255, 0)  # Зелёный цвет для боксов
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            # Формируем текст для отображения
            display_text = ""

            # Добавляем score если есть
            if 'score' in ann:
                display_text += f"{ann['score']:.2f}"

            # Добавляем категорию если есть
            category_id = ann['category_id']
            category = next((cat for cat in self.coco_data['categories'] if cat['id'] == category_id), None)
            if category:
                if display_text:
                    display_text += " "
                display_text += category['name']

            # Рисуем текст
            if display_text:
                cv2.putText(image, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw segmentation if exists
            if 'segmentation' in ann:
                for seg in ann['segmentation']:
                    points = np.array(seg, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(image, [points], True, (0, 0, 255), 2)

        return image

    def show_current_image(self):
        if not self.image_files:
            print("No images found in the folder")
            return

        current_file = self.image_files[self.current_index]
        image_path = os.path.join(self.image_folder, current_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load image: {image_path}")
            return

        # image_info = self.filename_to_image_info.get(current_file, None)
        image_info = self.filename_to_image_info.get(current_file.strip('0'), None) #### УДАЛИТЬ ЛИШНИЕ НУЛИ В НАЧАЛЕ
        annotated_image = self.draw_annotations(image.copy(), image_info)
        print(image_info)

        # Add status text
        status_text = f"Image {self.current_index + 1}/{len(self.image_files)}: {current_file}"
        cv2.putText(annotated_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 0)

        # Add threshold info
        threshold_text = f"Conf threshold: {self.conf_threshold:.2f} (W/S to adjust)"
        cv2.putText(annotated_image, threshold_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0, (0, 0, 0), 2)
        cv2.putText(annotated_image, threshold_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 0)

        # Add help text
        help_text = "A/D: navigate, W/S: conf threshold, ESC: exit"
        cv2.putText(annotated_image, help_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 0)

        cv2.imshow(self.window_name, annotated_image)

    def run(self):
        if not self.image_files:
            print("No images found in the specified folder")
            return

        self.show_current_image()

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == 27:  # ESC
                break
            elif key == ord('a') or key == ord('A'):  # Previous image
                self.current_index = max(0, self.current_index - 1)
                self.show_current_image()
            elif key == ord('d') or key == ord('D'):  # Next image
                self.current_index = min(len(self.image_files) - 1, self.current_index + 1)
                self.show_current_image()
            elif key == ord('w') or key == ord('W'):  # Increase threshold
                self.conf_threshold = min(0.99, self.conf_threshold + 0.01)
                self.show_current_image()
            elif key == ord('s') or key == ord('S'):  # Decrease threshold
                self.conf_threshold = max(-0.01, self.conf_threshold - 0.01)
                self.show_current_image()

        cv2.destroyAllWindows()





if __name__ == "__main__":
    # import argparse
    #
    # parser = argparse.ArgumentParser(description="COCO JSON Viewer")
    # parser.add_argument("image_folder", help="Path to folder containing images")
    # parser.add_argument("coco_json", help="Path to COCO format JSON file")
    #
    # args = parser.parse_args()
    #
    # viewer = CocoViewer(args.image_folder, args.coco_json)
    # viewer.run()

    # image_folder = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\vid1\images'
    # coco_json = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\vid1\vid1.json'
    # coco_json = pred_pathes['vid1_drone FOMO_56_42e_res_v0_focal']


    image_folder = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test_nobird\images'
    coco_json = gt_pathes['skb_test_nobird']



    # coco_json = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\submit\vid1\coco_predictions.json'
    # coco_json = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\runs\detect\val y=12n p=1088 vid1_imgs5\coco_predictions.json'

    viewer = CocoViewer(image_folder, coco_json)
    viewer.run()

    ### VAL
    # image_folder = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\val\images'
    # coco_json = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\annotations\split_val_coco.json'

    ### FOMO
    # image_folder = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train_FOMO\val\images'
    # coco_json = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train_FOMO\mva23_FOMO_val.json'
    # coco_json = 'FOMO_50e_predictions.json'

    ### FOMO
    # image_folder = r'C:\Users\ILYA\.clearml\cache\storage_manager\datasets\ds_ae8c12c33b324947af9ae6379d920eb8\images\train'
    # coco_json = r'C:\Users\ILYA\.clearml\cache\storage_manager\datasets\ds_ae8c12c33b324947af9ae6379d920eb8\train_annotations.json'


    ## FOMO bg | dronesonly FOMO val
    # image_folder = r'C:\Users\ILYA\.clearml\cache\storage_manager\datasets\ds_45062c8b1fac490480d105ad9c945f22\val\images'
    # coco_json = pred_pathes['drones_only_FOMO_val FOMO_bg_56_14e']
    image_folder = r'C:\Users\ILYA\.clearml\cache\storage_manager\datasets\ds_ae8c12c33b324947af9ae6379d920eb8\images\val'
    # coco_json = pred_pathes['drones_only_val FOMO_bg_56_14e']
    coco_json = pred_pathes['drones_only_val FOMO_56_104e']
    coco_json = r'predictions\drones_only!BEST_FOMO_56_crossEntropy_drones_only_FOMO_1.0.1_14e_model_weights\annotations\predictions_aligned.json'
    coco_json = r'predictions\NORESIZE_drones_only!BEST_FOMO_56_crossEntropy_dronesOnly_104e_model_weights\annotations\predictions.json'
    # coco_json = gt_pathes['drones_only_val']


    ### baseline val
    # image_folder = r'C:\Users\ILYA\.clearml\cache\storage_manager\datasets\ds_ae8c12c33b324947af9ae6379d920eb8\images\val'
    # coco_json = pred_pathes['drones_only_val baseline']
    # coco_json = 'predictions/baseline/baseline_dronesonly_val.json'

    #baseline hardnegative
    # image_folder = r'C:\Users\ILYA\.clearml\cache\storage_manager\datasets\ds_ae8c12c33b324947af9ae6379d920eb8\images\train'
    # coco_json = 'predictions/baseline/train_coco_hard_negative.json'


    # image_folder = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\vid2\screens'
    # coco_json = os.path.join(image_folder, 'FOMO_50e_predictions.json')


    # old mva23 bird val prdict
    # image_folder = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\val\images'
    # coco_json = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\submit\mva23_val\coco_predictions.json'
    # coco_json = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\submit\mva23_val\predicts_new_script.json' # через мой скрипт


    viewer = CocoViewer(image_folder, coco_json)
    viewer.run()