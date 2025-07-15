FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

RUN apt update \
    && apt install -y sudo

# Install dependencies
RUN apt install -y git unzip wget xz-utils build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev curl software-properties-common libbz2-dev liblzma-dev libsqlite3-dev

# Install git-lfs
RUN apt install -y docker.io \
    && apt-get install -y git-lfs \
    && git-lfs install

# Install vim
RUN apt install -y vim

# Install zsh
RUN apt install -y zsh \
    && git clone https://github.com/ohmyzsh/ohmyzsh.git ~/.oh-my-zsh \
    && cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc \
    && sed -i "s/robbyrussell/ys/" ~/.zshrc

# Install zsh-syntax-highlighting
RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ~/zsh-syntax-highlighting \ 
    && echo "source ~/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" >> ~/.zshrc

# Install zsh-autosuggestions
RUN git clone https://github.com/zsh-users/zsh-autosuggestions ~/.zsh/zsh-autosuggestions \
    && echo "source ~/.zsh/zsh-autosuggestions/zsh-autosuggestions.zsh" >> ~/.zshrc

# Install python
RUN wget https://www.python.org/ftp/python/3.8.19/Python-3.8.19.tar.xz \
    && tar -xf Python-3.8.19.tar.xz \
    && cd Python-3.8.19 \
    && ./configure --enable-loadable-sqlite-extensions \
    && sudo make altinstall \
    && echo "alias python3='/usr/local/bin/python3.8'" >> ~/.zshrc \
    && echo "alias python='/usr/local/bin/python3.8'" >> ~/.zshrc 

RUN /usr/local/bin/python3.8 -m pip install --upgrade pip wheel \
    && /usr/local/bin/python3.8 -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 \
    && /usr/local/bin/python3.8 -m pip install accelerate==0.19.0 datasets==2.12.0 decord==0.6.0 deepspeed==0.9.3 diffusers==0.17.0 einops==0.6.1 ftfy==6.1.1 gradio==3.44.0 imageio==2.31.1 iopath==0.1.10 ipdb==0.13.13 joblib==1.3.1 matplotlib==3.7.1 mdtex2html==1.2.0 numpy==1.24.3 packaging==23.1 pandas==2.0.2 peft==0.3.0 Pillow==9.5.0 pytorchvideo==0.1.5 PyYAML==6.0 regex==2023.6.3 scipy timm==0.9.2 tqdm==4.65.0 transformers==4.29.2 omegaconf==2.3.0 tensorboard==2.13.0 sentencepiece \
    && /usr/local/bin/python3.8 -m pip install setuptools==69.5.1

# Support loading mp3 files by torchaudio
RUN apt install -y sox \
    && add-apt-repository -y ppa:savoury1/ffmpeg4 \
    && apt-get -qq install -y ffmpeg

RUN echo "cd $HOME" >> ~/.zshrc
ENTRYPOINT ["/bin/zsh"]

