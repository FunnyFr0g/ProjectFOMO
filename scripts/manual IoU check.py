import os
import json
import cv2
import numpy as np
from matplotlib import cm


class BBoxVisualizer:
    def __init__(self, gt_json, pred_json, image_dir, iou_thresholds=[0.5, 0.75]):
        self.gt_json = gt_json
        self.pred_json = pred_json
        self.image_dir = image_dir
        self.iou_thresholds = iou_thresholds

        # Load annotations
        self.gt_data = self._load_json(gt_json)
        self.pred_data = self._load_json(pred_json)

        # Create image_id to filename mapping
        self.image_id_to_info = {img['id']: img for img in self.gt_data['images']}

        # Organize annotations by image_id
        self.gt_anns = self._organize_annotations(self.gt_data)
        self.pred_anns = self._organize_annotations(self.pred_data)

        # Get all image ids
        self.image_ids = list(self.gt_anns.keys())
        self.current_image_idx = 0
        self.score_threshold = 0.5

        # Colormap for IoU-based coloring
        self.colormap = cm.get_cmap('viridis')

        # Statistics
        self.stats = {}

    def _load_json(self, json_path):
        with open(json_path, 'r') as f:
            return json.load(f)

    def _organize_annotations(self, data):
        anns_by_image = {}
        for ann in data['annotations']:
            image_id = ann['image_id']
            if image_id not in anns_by_image:
                anns_by_image[image_id] = []
            anns_by_image[image_id].append(ann)
        return anns_by_image

    def _calculate_iou(self, box1, box2):
        # box format: [x, y, width, height]
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate coordinates of intersection rectangle
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2

        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou

    def _find_best_match(self, pred_box, gt_boxes):
        best_iou = 0
        best_idx = -1

        for i, gt_box in enumerate(gt_boxes):
            iou = self._calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        return best_idx, best_iou

    def _calculate_metrics(self, image_id):
        gt_boxes = [ann['bbox'] for ann in self.gt_anns.get(image_id, [])]
        pred_boxes = [ann['bbox'] for ann in self.pred_anns.get(image_id, [])
                      if 'score' in ann and ann['score'] >= self.score_threshold]

        # Initialize metrics
        metrics = {
            'gt_count': len(gt_boxes),
            'dt_count': len(pred_boxes),
            'tp': {thresh: 0 for thresh in self.iou_thresholds},
            'fp': {thresh: 0 for thresh in self.iou_thresholds},
            'fn': {thresh: 0 for thresh in self.iou_thresholds},
            'ious': {},
            'image_info': self.image_id_to_info[image_id],
            'gt_anns': self.gt_anns.get(image_id, []),
            'pred_anns': [ann for ann in self.pred_anns.get(image_id, [])
                          if 'score' in ann and ann['score'] >= self.score_threshold]
        }

        matched_gt = [False] * len(gt_boxes)

        # For each prediction, find best matching GT
        for i, pred_box in enumerate(pred_boxes):
            best_gt_idx, best_iou = self._find_best_match(pred_box, gt_boxes)
            metrics['ious'][i] = best_iou

            if best_gt_idx != -1:
                for thresh in self.iou_thresholds:
                    if best_iou >= thresh:
                        if not matched_gt[best_gt_idx]:
                            metrics['tp'][thresh] += 1
                            matched_gt[best_gt_idx] = True
                        else:
                            metrics['fp'][thresh] += 1
                    else:
                        metrics['fp'][thresh] += 1
            else:
                for thresh in self.iou_thresholds:
                    metrics['fp'][thresh] += 1

        # Calculate FN (unmatched GTs)
        for gt_matched in matched_gt:
            if not gt_matched:
                for thresh in self.iou_thresholds:
                    metrics['fn'][thresh] += 1

        return metrics

    def _draw_bbox(self, image, box, color, label, iou=None, score=None):
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        # Create label text
        label_text = label
        if iou is not None:
            label_text += f" IoU:{iou:.4f}"
        if score is not None:
            label_text += f" Score:{score:.4f}"

        # Draw label background
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x, y - text_height - 4), (x + text_width, y), color, -1)

        # Draw label text
        cv2.putText(image, label_text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def _draw_metrics(self, image, metrics):
        y_offset = 20
        line_height = 25

        # Draw general info
        cv2.putText(image, f"GT: {metrics['gt_count']} | DT: {metrics['dt_count']}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += line_height

        # Draw metrics for each threshold
        for thresh in self.iou_thresholds:
            cv2.putText(image,
                        f"IoU@{thresh:.4f}: TP={metrics['tp'][thresh]} FP={metrics['fp'][thresh]} FN={metrics['fn'][thresh]}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += line_height

        # Draw current score threshold
        cv2.putText(image, f"Score threshold: {self.score_threshold:.4f} (W/S to adjust)",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def _print_image_info(self, metrics):
        print("\n" + "=" * 80)
        print(f"Image: {metrics['image_info']['file_name']}")
        print(f"Dimensions: {metrics['image_info']['width']}x{metrics['image_info']['height']}")
        print(f"GT boxes: {metrics['gt_count']}")
        print(f"Detections (with score >= {self.score_threshold:.2f}): {metrics['dt_count']}")

        for thresh in self.iou_thresholds:
            print(f"\nMetrics @ IoU {thresh:.2f}:")
            print(f"  TP: {metrics['tp'][thresh]}")
            print(f"  FP: {metrics['fp'][thresh]}")
            print(f"  FN: {metrics['fn'][thresh]}")

        print("\nGround Truth boxes:")
        for i, ann in enumerate(metrics['gt_anns']):
            print(f"  GT {i + 1}: bbox={ann['bbox']}= category_id={ann['category_id']}")

        print("\nDetection boxes:")
        for i, ann in enumerate(metrics['pred_anns']):
            iou = metrics['ious'].get(i, 0)
            print(f"  DT {i + 1}: bbox={ann['bbox']}= score={ann['score']:.4f} IoU={iou:.4f}")

        print("=" * 80 + "\n")

    def visualize(self):
        cv2.namedWindow("BBox Visualizer", cv2.WINDOW_NORMAL)

        while True:
            image_id = self.image_ids[self.current_image_idx]
            image_info = self.image_id_to_info[image_id]
            image_path = os.path.join(self.image_dir, image_info['file_name'])

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                return

            # Calculate metrics
            metrics = self._calculate_metrics(image_id)
            self.stats[image_id] = metrics

            # Print info to console
            self._print_image_info(metrics)

            # Draw GT boxes (white)
            for ann in metrics['gt_anns']:
                self._draw_bbox(image, ann['bbox'], (255, 255, 255), "GT")

            # Draw predicted boxes (color by IoU)
            for i, ann in enumerate(metrics['pred_anns']):
                iou = metrics['ious'].get(i, 0)
                # Map IoU to color (0=red, 1=green)
                color_rgb = [int(255 * x) for x in self.colormap(iou)[:3]]
                color_bgr = color_rgb[::-1]  # OpenCV uses BGR

                self._draw_bbox(image, ann['bbox'], color_bgr, "DT",
                                iou=iou, score=ann.get('score', None))

            # Draw metrics
            self._draw_metrics(image, metrics)

            # Update window title with image name
            cv2.setWindowTitle("BBox Visualizer", f"BBox Visualizer - {image_info['file_name']}")

            # Show image
            cv2.imshow("BBox Visualizer", image)

            # Handle key presses
            key = cv2.waitKey(0) & 0xFF
            if key == ord('a'):  # Previous image
                self.current_image_idx = max(0, self.current_image_idx - 1)
            elif key == ord('d'):  # Next image
                self.current_image_idx = min(len(self.image_ids) - 1, self.current_image_idx + 1)
            elif key == ord('w'):  # Increase score threshold
                self.score_threshold = min(1.0, round(self.score_threshold + 0.05, 2))
                print(f"\nScore threshold increased to {self.score_threshold:.4f}")
            elif key == ord('s'):  # Decrease score threshold
                self.score_threshold = max(0.0, round(self.score_threshold - 0.05, 2))
                print(f"\nScore threshold decreased to {self.score_threshold:.4f}")
            elif key == 27:  # ESC to exit
                break

        cv2.destroyAllWindows()





# Example usage
if __name__ == "__main__":
    from label_pathes import gt_pathes, pred_pathes

    ### SKB
    # gt_json = gt_pathes['skb_test']
    # pred_json = pred_pathes['skb_test YOLO12n 1088px']
    # image_dir = r"X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test\images"

    ### FOMO val
    gt_json = gt_pathes['mva23_val_FOMO']
    pred_json = pred_pathes['mva23_val_FOMO FOMO 50e']
    image_dir = r"X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train_FOMO\val\images"


    # You can adjust these thresholds
    iou_thresholds = [0.1, 0.5]

    visualizer = BBoxVisualizer(gt_json, pred_json, image_dir, iou_thresholds)
    visualizer.visualize()