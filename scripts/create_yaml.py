import yaml

data_yaml = {
    'train': r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\train',
    'val': r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\val',
    'names': {0: 'bird'},
    'nc': 1
}

with open(r'X:\SOD\MVA2023SmallObjectDetection4SpottingBirds\data\mva2023_sod4bird_train\data.yaml', 'w') as f:
    yaml.dump(data_yaml, f)