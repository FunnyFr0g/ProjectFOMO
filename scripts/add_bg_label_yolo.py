import os

data_dir = r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\vid1'
img_path = data_dir+r'\images'
lb_path = data_dir+r'\labels'

images = os.listdir(img_path)
labels = os.listdir(lb_path)

print(labels)



for img in images:
    lb = img.strip('.jpg')+'.txt'
    if lb not in labels:
        print(lb + ' не в файлах')
        with open(lb_path+'\\'+lb, 'w'):
            pass

