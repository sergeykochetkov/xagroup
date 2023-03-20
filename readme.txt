conda create --name openmmlab python=3.8 -y

conda install pytorch torchvision -c pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install openmim
mim install mmcv-full
mim install mmdet
mim install mmsegmentation

git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc6
pip install -e .


conda create -n yolo3d python=3.8 numpy
conda activate yolo3d
pip install torch==1.8.1 torchvision==0.9.1
cd YOLO3D
pip install -r requirements.txt
