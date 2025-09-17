import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random as r

r.seed(0)

iou_th = 0.7
n = 100

gt=[i%2 for i in range(n )]


iou = [r.normalvariate(0.6, 0.2) for _ in range(n)]
print(f'{iou=}')
# scores = np.linspace(0.0, 1, n)
scores = [r.uniform(0, 1) for _ in range(n)]

# y_true=np.array([1 if m>iou_th else 0 for m in iou])
y_true=np.array([1 if m<n*0.75 else 0 for m in range(0,n)])

print(f'{y_true=}')

fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1, drop_intermediate=False)
roc_auc = auc(fpr, tpr)

print(f'{fpr=}')
print(f'{tpr=}')
print(f'{thresholds=}')
print(f'{len(fpr)=}')

def plot_roc_curve(fpr, tpr, roc_auc):
    """Визуализация ROC кривой"""
    title = f'sbox'

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title('ROC '+title)
    plt.legend(loc="lower right")
    plt.savefig(r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\ROC'+'\\'+title +'.png')
    plt.show()

plot_roc_curve(fpr, tpr, roc_auc)