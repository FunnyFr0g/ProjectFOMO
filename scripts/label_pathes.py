import os
gt_pathes = {'vid1':r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\vid1\vid1.json',
                'mva23_val':r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\val\mva23_val_UPDATED.json',
               'skb_test':r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test\skb_test.json',
             'mva23_val_FOMO': r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train_FOMO\mva23_FOMO_val.json'
                   }



pred_pathes = {'vid1 YOLO12n 640px':r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\YOLO\runs\detect\val y=12n p=640 vid1\coco_predictions.json',
                 'vid1 YOLO12n 1088px': r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\YOLO\runs\detect\val y=12n p=1088 vid1\coco_predictions.json',
                 'vid1 baseline':r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\submit\vid1\coco_predictions.json',
                 'mva23_val YOLO12n 640px': r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\YOLO\runs\detect\val y=12n p=640 mva23_val\coco_predictions.json',
                 'mva23_val YOLO12n 1088px': r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\YOLO\runs\detect\val y=12n p=1088 mva23_val\coco_predictions.json',
                 'mva23_val baseline': r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\submit\mva23_val\coco_predictions.json',
                 'skb_test YOLO12n 640px': r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\YOLO\runs\detect\val y=12n p=640 skb_test\coco_predictions.json',
                 'skb_test YOLO12n 1088px': r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\YOLO\runs\detect\val y=12n p=1088 skb_test\coco_predictions.json',
                 'skb_test baseline': r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\submit\skb_test\coco_predictions.json',
                  'mva23_val_FOMO FOMO 50e':  r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train_FOMO\val\FOMO_50e_predictions.json',
                  'mva23_val_FOMO FOMO 50e SAHI 800p':  r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train_FOMO\val\FOMO_50e_SAHI_800p_predictions_tiled.json',
                  'mva23_val_FOMO FOMO 10e': r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train_FOMO\val\FOMO_50e_predictions.json',  #'FOMO_10e_predictions.json',
                'skb_test FOMO 50e' : r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test\FOMO_50e_predictions.json',
                'skb_test FOMO 50e no_resize' : r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test\FOMO_50e_predictions_no_resize.json',
                'skb_test FOMO112 50e' : r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test\FOMO_112p_50e_predictions.json',
                'skb_test FOMO112 10e' : r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test\FOMO_112p_10e_predictions.json',
               'skb_test FOMO 50e SAHI 448p' : r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test\FOMO_50e_SAHI_448p_predictions_tiled.json',
               'skb_test FOMO 50e SAHI 640p' : r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test\FOMO_50e_SAHI_640p_predictions_tiled.json',
               'skb_test FOMO 50e SAHI 800p' : r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\skb_test\FOMO_50e_SAHI_800p_predictions_tiled.json',
               'vid1 FOMO 50e' : r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\vid1\FOMO_50e_predictions.json',
               'vid1 FOMO 50e SAHI 800p': r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\vid1\FOMO_50e_SAHI_800p_predictions_tiled.json',
               }



if __name__ == '__main__':
    for n, p in gt_pathes.items():
        if os.path.exists(p):
            print(f'Успешно {n}')
        else:
            print(f'Косяк для {n}')

    for n, p in pred_pathes.items():
        if os.path.exists(p):
            print(f'Успешно {n}')
        else:
            print(f'Косяк для {n}')