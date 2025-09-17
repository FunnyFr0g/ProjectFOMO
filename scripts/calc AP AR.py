import json
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from label_pathes import gt_pathes, pred_pathes

# Configuration
iou_thresholds = [0.1, 0.5, 0.9]  # Пример массива IoU thresholds


# Initialize results storage
results = {iou: {} for iou in iou_thresholds}

for iou in iou_thresholds:
    for dataset_name in gt_pathes.keys():
        # Load ground truth
        coco_gt = COCO(gt_pathes[dataset_name])

        # Initialize dataset entry in results
        results[iou][dataset_name] = {}

        # Check all models for this dataset
        for model_key in pred_pathes.keys():
            if dataset_name in model_key:
                model_name = model_key.replace(f"{dataset_name} ", "")

                # Load predictions
                with open(pred_pathes[model_key], 'r') as f:
                    coco_pred = json.load(f)

                # Convert predictions to COCO format if needed
                if isinstance(coco_pred, dict) and 'annotations' in coco_pred:
                    coco_pred = coco_pred['annotations']

                # Evaluate
                coco_dt = coco_gt.loadRes(coco_pred)
                coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

                # Set IoU threshold
                coco_eval.params.iouThrs = np.array([iou])

                # Evaluate
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

                # Store results
                results[iou][dataset_name][model_name] = {
                    'AP': coco_eval.stats[0],
                    'AP50': coco_eval.stats[1],
                    'AP75': coco_eval.stats[2],
                    'APs': coco_eval.stats[3],
                    'APm': coco_eval.stats[4],
                    'APl': coco_eval.stats[5],
                    'AR1': coco_eval.stats[6],
                    'AR10': coco_eval.stats[7],
                    'AR100': coco_eval.stats[8],
                    'ARs': coco_eval.stats[9],
                    'ARm': coco_eval.stats[10],
                    'ARl': coco_eval.stats[11]
                }

    # Save metrics for this IoU threshold to CSV
    df_list = []
    for dataset_name in results[iou]:
        df = pd.DataFrame.from_dict(results[iou][dataset_name], orient='index')
        df['dataset'] = dataset_name
        df_list.append(df)

    combined_df = pd.concat(df_list)
    combined_df = combined_df.reset_index().rename(columns={'index': 'model'})

    # Pivot table for better readability
    pivot_ap = combined_df.pivot(index='dataset', columns='model', values='AP')
    pivot_ar = combined_df.pivot(index='dataset', columns='model', values='AR100')

    # Save to CSV
    iou_str = str(int(iou * 100)).zfill(2)
    pivot_ap.to_csv(f'metrics_iou={iou_str}_AP.csv')
    pivot_ar.to_csv(f'metrics_iou={iou_str}_AR.csv')

print("Evaluation completed. Results saved to CSV files.")