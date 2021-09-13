# Set up Google Research Football environment
# https://github.com/google-research/football/blob/master/Dockerfile

FROM tensorflow/tensorflow:1.15.2-gpu-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get --no-install-recommends install -yq git cmake build-essential \
  libgl1-mesa-dev libsdl2-dev \
  libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
  libdirectfb-dev libst-dev mesa-utils xvfb x11vnc \
  libsdl-sge-dev python3-pip

RUN python3 -m pip install --upgrade pip setuptools psutil gfootball \
 dm-sonnet==1.* git+https://github.com/openai/baselines.git@master

# Set up api environment

RUN apt-get install -y vim tmux
RUN python3 -m pip install --upgrade jupyterlab flask waitress pytest

WORKDIR /app
#COPY api ./api
#COPY tests ./tests

