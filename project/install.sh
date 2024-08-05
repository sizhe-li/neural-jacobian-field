# python=3.9.18
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install git+https://github.com/sizhe-li/nerfstudio.git
pip install numpy==1.26.4 --ignore-installed
pip install opencv-python==4.8.0.74 --ignore-installed
pip install -r requirements.txt