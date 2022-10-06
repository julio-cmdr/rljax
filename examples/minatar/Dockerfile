FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# declare the image name
ENV IMG_NAME=10.1-cudnn7-devel-ubuntu18.04 \
    # declare what jaxlib tag to use
    # if a CI/CD system is expected to pass in these arguments
    # the dockerfile should be modified accordingly
    JAXLIB_VERSION=0.1.69

# solve bugs in apt keys: https://github.com/NVIDIA/nvidia-docker/issues/1631#issuecomment-1112828208
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# install python3-pip
RUN apt -y update
RUN apt -y install python3.8
RUN apt -y install wget
RUN apt -y install python3-pip
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.8 get-pip.py

# install git
RUN apt -y install git

# install jax via pip
RUN python3.8 -m pip install jax==0.2.25 https://storage.googleapis.com/jax-releases/cuda101/jaxlib-${JAXLIB_VERSION}+cuda101-cp38-none-manylinux2010_x86_64.whl

# install MinAtar dependencies via pip
RUN git clone https://github.com/kenjyoung/MinAtar
RUN python3.8 -m pip install MinAtar/
RUN yes | rm -r MinAtar

# install rljax dependencies via pip
RUN python3.8 -m pip install gym==0.17.2 dm-haiku==0.0.5 optax==0.1.0 opencv-python==4.5.4.60 tensorboardX==2.1 tqdm==4.48.2

# install opencv dependencies
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Clone rljax
RUN git clone https://github.com/julio-cmdr/rljax
