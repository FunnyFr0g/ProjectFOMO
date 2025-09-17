import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import shutil
from pycocotools.coco import COCO
from typing import Dict, List, Tuple

random.seed(42)


class COCOBboxResizer:
    def __init__(
        self,
        coco_annotation_path: str,
        image_dir: str,
        output_dir: str,
        min_bbox: int,
        max_bbox: int,
        target_image_size: Tuple[int, int] = (640, 640),
        jitter: float = 0.2,
        need_masks=True
    ):
        """
        Args:
            coco_annotation_path: Path to COCO annotation file
            image_dir: Directory with images
            output_dir: Directory to save processed images and annotations
            min_bbox: Minimum bbox size (in pixels) after processing
            max_bbox: Maximum bbox size (in pixels) after processing
            target_image_size: Size of output images (width, height)
            jitter: Relative jitter for crop position (0-1)
        """
        self.coco_annotation_path = coco_annotation_path
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.min_bbox = min_bbox
        self.max_bbox = max_bbox
        self.target_width, self.target_height = target_image_size
        self.jitter = jitter
        self.need_masks = need_masks

        # Create output directories
        self.output_image_dir = os.path.join(output_dir, "images")
        self.output_mask_dir = os.path.join(output_dir, "fomo_masks")
        os.makedirs(self.output_image_dir, exist_ok=True)
        os.makedirs(self.output_mask_dir, exist_ok=True)

        # Load original COCO dataset
        self.coco = COCO(coco_annotation_path)
        self.original_images = self.coco.dataset["images"]
        self.original_annotations = self.coco.dataset["annotations"]
        self.original_categories = self.coco.dataset["categories"]

        # Prepare new COCO dataset
        self.new_dataset = {
            "images": [],
            "annotations": [],
            "categories": self.original_categories,
        }
        self.annotation_id = 1

    def _calculate_scale_factor(self, bbox_width: float, bbox_height: float) -> float:
        """Calculate scale factor to resize bbox to target size range."""
        bbox_size = max(bbox_width, bbox_height)
        target_size = random.uniform(self.min_bbox, self.max_bbox)
        return target_size / bbox_size

    def _get_random_crop_position(
        self, bbox: List[float], scaled_width: int, scaled_height: int
    ) -> Tuple[int, int]:
        """Calculate random crop position with jitter around the bbox center."""
        bbox_center_x = bbox[0] + bbox[2] / 2
        bbox_center_y = bbox[1] + bbox[3] / 2

        # Convert bbox center to scaled image coordinates
        scaled_center_x = bbox_center_x * self.scale_factor
        scaled_center_y = bbox_center_y * self.scale_factor

        # Calculate max jitter distance
        max_jitter_x = self.jitter * self.target_width
        max_jitter_y = self.jitter * self.target_height

        # Calculate crop position with jitter
        crop_x = max(
            0,
            min(
                scaled_center_x - self.target_width / 2 + random.uniform(-max_jitter_x, max_jitter_x),
                scaled_width - self.target_width,
            ),
        )
        crop_y = max(
            0,
            min(
                scaled_center_y - self.target_height / 2 + random.uniform(-max_jitter_y, max_jitter_y),
                scaled_height - self.target_height,
            ),
        )

        return int(crop_x), int(crop_y)

    def _create_fomo_mask(
        self, annotations: List[Dict], crop_x: int, crop_y: int, image_id: int
    ) -> None:
        """Create FOMO mask (8-bit grayscale where pixel value = class_id)."""
        mask = Image.new("L", (self.target_width, self.target_height), 0)
        draw = ImageDraw.Draw(mask)

        for ann in annotations:
            if ann["image_id"] != image_id:
                continue

            # Scale and shift bbox
            x1 = ann["bbox"][0] * self.scale_factor - crop_x
            y1 = ann["bbox"][1] * self.scale_factor - crop_y
            x2 = x1 + ann["bbox"][2] * self.scale_factor
            y2 = y1 + ann["bbox"][3] * self.scale_factor

            # Clip to image bounds
            x1 = max(0, min(x1, self.target_width))
            y1 = max(0, min(y1, self.target_height))
            x2 = max(0, min(x2, self.target_width))
            y2 = max(0, min(y2, self.target_height))

            if x1 >= x2 or y1 >= y2:
                continue

            # Draw rectangle with class_id as pixel value
            draw.rectangle([x1, y1, x2, y2], fill=ann["category_id"]+1)

        mask.save(os.path.join(self.output_mask_dir, f"{image_id}.png"))

    def process_image(self, image_info: Dict) -> None:
        """Process single image."""
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        image_id = image_info["id"]

        # Load image
        img = Image.open(image_path).convert("RGB")
        original_width, original_height = img.size

        # Get all annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        if not annotations:
            return

        # Select a random annotation to determine scale
        target_ann = random.choice(annotations)
        bbox = target_ann["bbox"]

        # Calculate scale factor based on the largest dimension of the bbox
        self.scale_factor = self._calculate_scale_factor(bbox[2], bbox[3])

        # Scale image
        scaled_width = int(original_width * self.scale_factor)
        scaled_height = int(original_height * self.scale_factor)
        scaled_img = img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)

        # Calculate random crop position
        crop_x, crop_y = self._get_random_crop_position(bbox, scaled_width, scaled_height)

        # Crop image
        cropped_img = scaled_img.crop(
            (
                crop_x,
                crop_y,
                crop_x + self.target_width,
                crop_y + self.target_height,
            )
        )

        # Save processed image
        new_image_name = f"{image_id}.jpg"
        cropped_img.save(os.path.join(self.output_image_dir, new_image_name))

        # Create FOMO mask
        if self.need_masks:
            self._create_fomo_mask(annotations, crop_x, crop_y, image_id)

        # Prepare new image info for COCO dataset
        new_image_info = {
            "id": image_id,
            "file_name": new_image_name,
            "width": self.target_width,
            "height": self.target_height,
        }
        self.new_dataset["images"].append(new_image_info)

        # Process and add annotations
        for ann in annotations:
            # Scale and shift bbox
            x1 = ann["bbox"][0] * self.scale_factor - crop_x
            y1 = ann["bbox"][1] * self.scale_factor - crop_y
            w = ann["bbox"][2] * self.scale_factor
            h = ann["bbox"][3] * self.scale_factor

            # Clip to image bounds
            x1 = max(0, min(x1, self.target_width))
            y1 = max(0, min(y1, self.target_height))
            w = max(0, min(w, self.target_width - x1))
            h = max(0, min(h, self.target_height - y1))

            if w <= 0 or h <= 0:
                continue

            new_ann = {
                "id": self.annotation_id,
                "image_id": image_id,
                "category_id": ann["category_id"]+1, ######### ВАЖНО
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": ann["iscrowd"],
            }
            self.new_dataset["annotations"].append(new_ann)
            self.annotation_id += 1

    def process_dataset(self) -> None:
        """Process all images in the dataset."""
        for image_info in tqdm(self.original_images, desc="Processing images"):
            try:
                self.process_image(image_info)
            except Exception as e:
                print(f"Error processing image {image_info['id']}: {e}")

        # Save new COCO annotations
        output_annotation_path = os.path.join(self.output_dir, "annotations.json")
        with open(output_annotation_path, "w") as f:
            json.dump(self.new_dataset, f)

        print(f"Processing complete. Results saved to {self.output_dir}")


# Example usage
if __name__ == "__main__":
    # Configuration
    coco_annotation_path = r"X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\annotations\split_train_coco.json"
    image_dir = r"X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\train\images"
    output_dir = r"X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train_FOMO"
    min_bbox = 4  # Minimum bbox size in pixels
    max_bbox = 16  # Maximum bbox size in pixels
    target_image_size = (224, 224)  # Output image size
    jitter = 0.2  # Relative jitter for crop position

    # Create and run processor
    processor = COCOBboxResizer(
        coco_annotation_path=coco_annotation_path,
        image_dir=image_dir,
        output_dir=output_dir,
        min_bbox=min_bbox,
        max_bbox=max_bbox,
        target_image_size=target_image_size,
        jitter=jitter,
        need_masks=False
    )
    processor.process_dataset()