import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops.misc import interpolate

from label_pathes import gt_pathes, pred_pathes
import os
from sklearn.metrics import auc as sk_auc

def plot_precision_recall_curve(precision, recall, ap, title):
    """Визуализирует Precision-Recall кривую"""
    dataset_name, model_name = title.split()
    new_model_name = model_name.replace(' ', '_')
    # plt.figure()
    plt.plot(recall, precision, label=f'AP={ap:.2f} {model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(dataset_name)
    plt.legend(loc="lower left", fontsize=11)
    # plt.savefig(save_path)
    # plt.close()



PR_path = r'C:\Users\ILYA\PycharmProjects\PythonProject\model_graphics\PR arrays'
datasets_list = ['drones_only_FOMO_val', 'drones_only_val', 'vid1_drone']
    # model_name_list = ['FOMO_56_104e','FOMO_56_104e_NORESIZE', 'FOMO_bg_56_14e','FOMO_56_22e_bg_crop', 'FOMO_56_35e_res_v0', 'baseline', ]
    # model_name_list = ['FOMO_56_104e', 'FOMO_bg_56_14e','FOMO_56_22e_bg_crop', 'FOMO_56_35e_res_v0',
    #                    'FOMO_56_42e_res_v1', 'FOMO_56_150e_res_v1', 'FOMO_56_42e_res_v1_focal','FOMO_56_71e_res_v1_focal', 'baseline', ]
model_name_list = ['FOMO_56_104e', 'FOMO_bg_56_14e','FOMO_56_22e_bg_crop', 'FOMO_56_35e_res_v0', 'FOMO_56_42e_res_v0_focal',
                   'FOMO_56_42e_res_v1', 'FOMO_56_150e_res_v1', 'FOMO_56_42e_res_v1_focal','FOMO_56_71e_res_v1_focal', 'baseline', ]

model_name_list = [
                   'FOMO_56_42e_res_v1', 'FOMO_56_150e_res_v1', 'FOMO_56_42e_res_v1_focal','FOMO_56_71e_res_v1_focal', 'baseline', ]


for dataset in datasets_list:
    for model_name in model_name_list:
        txt_path = os.path.join(PR_path, f'{dataset} {model_name}.txt')
        try:
            data = np.loadtxt(txt_path)
        except Exception as e:
            print(f'Для {dataset} {model_name} не найдено пути')
            print('!' * 20)
            print(e)
            continue
        precision = data[:, 0]
        recall = data[:, 1]
        ap = sk_auc(recall, precision)

        plot_precision_recall_curve(precision, recall, ap, f'{dataset} {model_name}')

    plt.show()
