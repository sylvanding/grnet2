#! /bin/bash

# GPU: 3080 Ti-12G
# PyTorch: 1.9.1
# cuda: 11.1
# python: 3.8

# venv
# conda create -n grnet python=3.8 -y
# conda activate grnet
# pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# cd /hy-tmp
# git clone --depth=1 -b main https://ghfast.top/https://github.com/sylvanding/grnet2 .
# chmod +x gpushare.sh
# oss login
# oss cp oss://mt_pc_16384_2048_10_15_5.5.tar .

tar -xvf mt_pc_16384_2048_10_15_5.5.tar # set the data path in the config file

# install opencv
sudo apt-get install -y libopencv-dev python3-opencv libpng-dev libjpeg-dev libtiff-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev

pip install -r requirements.txt

# pytorch3d - conda install the prebuilt version
# wget https://anaconda.org/pytorch3d/pytorch3d/0.6.1/download/linux-64/pytorch3d-0.6.1-py38_cu111_pyt191.tar.bz2
# conda install pytorch3d-0.6.1-py38_cu111_pyt191.tar.bz2 -y
# python -c "import torch; import pytorch3d; print(torch.__version__, pytorch3d.__version__);"

# pytorch3d - pip install from github and build from source
/usr/bin/python3.8 -m pip install --upgrade pip
git clone https://ghfast.top/https://github.com/facebookresearch/pytorch3d.git -b v0.6.1
cd pytorch3d
pip install .
# pip install git+https://ghfast.top/https://github.com/facebookresearch/pytorch3d.git@v0.6.1 # proxy
# pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.6.1
python -c "import torch; import pytorch3d; print(torch.__version__, pytorch3d.__version__);"

# install extensions
GRNET_HOME="/hy-tmp" # /repos/GRNet2

cd $GRNET_HOME/extensions/chamfer_dist
python setup.py install

# Cubic Feature Sampling
cd $GRNET_HOME/extensions/cubic_feature_sampling
python setup.py install

# Gridding & Gridding Reverse
cd $GRNET_HOME/extensions/gridding
python setup.py install

# Gridding Loss
cd $GRNET_HOME/extensions/gridding_loss
python setup.py install

# pointnet2_ops_lib
cd $GRNET_HOME/extensions/pointnet2_ops_lib
python setup.py install

# pointops
cd $GRNET_HOME/extensions/pointops
python setup.py install

# add extensions to PYTHONPATH - GRNet
echo 'export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.8/site-packages/chamfer-2.0.0-py3.8-linux-x86_64.egg' >> ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.8/site-packages/cubic_feature_sampling-1.1.0-py3.8-linux-x86_64.egg' >> ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.8/site-packages/gridding-2.1.0-py3.8-linux-x86_64.egg' >> ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.8/site-packages/gridding_distance-1.0.0-py3.8-linux-x86_64.egg' >> ~/.bashrc

# add extensions to PYTHONPATH - PointNet++
# DONE: add pointnet2_ops_lib and pointops
echo 'export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.8/site-packages/pointnet2_ops-3.0.0-py3.8-linux-x86_64.egg' >> ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.8/site-packages/pointops-0.0.0-py3.8-linux-x86_64.egg' >> ~/.bashrc

export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.8/site-packages/chamfer-2.0.0-py3.8-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.8/site-packages/cubic_feature_sampling-1.1.0-py3.8-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.8/site-packages/gridding-2.1.0-py3.8-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.8/site-packages/gridding_distance-1.0.0-py3.8-linux-x86_64.egg

# DONE: add pointnet2_ops_lib and pointops
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.8/site-packages/pointnet2_ops-3.0.0-py3.8-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.8/site-packages/pointops-0.0.0-py3.8-linux-x86_64.egg

ln -s /hy-tmp/output/logs /tf_logs
