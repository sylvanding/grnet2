#! /bin/bash

# GPU: 3080 Ti-12G
# PyTorch: 1.7.1
# cuda: 110
# python: 3.8

# cd /hy-tmp
# git clone --depth=1 -b scaled_3d_conv https://github.com/sylvanding/grnet2 .
# chmod +x gpushare.sh
# oss login
# oss cp oss://smlm_pc.tar .

tar -xvf smlm_pc.tar # set the data path in the config file

# install opencv
sudo apt-get install -y libopencv-dev python3-opencv libpng-dev libjpeg-dev libtiff-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev

pip install -r requirements.txt

# install extensions
GRNET_HOME="/hy-tmp"

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

# add extensions to PYTHONPATH
echo 'export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.8/site-packages/chamfer-2.0.0-py3.8-linux-x86_64.egg' >> ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.8/site-packages/cubic_feature_sampling-1.1.0-py3.8-linux-x86_64.egg' >> ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.8/site-packages/gridding-2.1.0-py3.8-linux-x86_64.egg' >> ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.8/site-packages/gridding_distance-1.0.0-py3.8-linux-x86_64.egg' >> ~/.bashrc
source ~/.bashrc

export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.8/site-packages/chamfer-2.0.0-py3.8-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.8/site-packages/cubic_feature_sampling-1.1.0-py3.8-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.8/site-packages/gridding-2.1.0-py3.8-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.8/site-packages/gridding_distance-1.0.0-py3.8-linux-x86_64.egg

ln -s /hy-tmp/output/logs /tf_logs
