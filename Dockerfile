# Set up Google Research Football environment
# https://github.com/google-research/football/blob/master/Dockerfile

FROM tensorflow/tensorflow:1.15.2-gpu-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get --no-install-recommends install -yq git cmake build-essential \
  libgl1-mesa-dev libsdl2-dev \
  libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
  libdirectfb-dev libst-dev mesa-utils xvfb x11vnc \
  libsdl-sge-dev python3-pip \
  vim tmux less wget


RUN python3 -m pip install --upgrade pip setuptools psutil \
 dm-sonnet==1.* git+https://github.com/openai/baselines.git@master \
 jupyterlab flask flask-cors waitress pytest torch kaggle-environments stringcase tqdm

# install modified google-research-football
RUN git clone https://github.com/IshikiFC/google-research-football.git /gfootball
RUN cd /gfootball && python3 -m pip install .

WORKDIR /app

