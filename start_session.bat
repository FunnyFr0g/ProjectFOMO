# запускается
clearml-session --store-workspace ~/mva23 --project SmallObjectDetection --docker nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 --packages "torch==1.12.1+cu118" "torchvision==0.13.1+cu118"

#не запускается
clearml-session  --python 3.9 --docker nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04  --store-workspace /home/user/bird_workspace --packages "torch==2.1.2" "torchvision==0.16.2" "torchaudio==2.1.2"

clearml-session --store-workspace ~/mva23 --project SmallObjectDetection --docker nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 --packages "torch==1.13.1+cu117"  "torchvision==0.14.1+cu117"
