import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Скачать только файлы из определенной папки
api.dataset_download_files(
    'klemenko/kitti-dataset',
    path='./data_object_image_2/testing',
    unzip=False,
    quiet=False
)