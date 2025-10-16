import os
import shutil
from pathlib import Path
from tqdm import tqdm

image_dir = Path(r'C:\Users\ILYA\Pictures\kitti\city_back')

file_names = os.listdir(image_dir)

for file_name in tqdm(file_names):
    if not file_name.startswith('background_'):
        new_filename = 'background_' + file_name
        os.rename(image_dir/file_name, image_dir/new_filename)
