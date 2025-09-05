FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        cmake \
        pkg-config \
        libgl1 \
        libglib2.0-0 \
        libxext6 \
        libsm6 \
        libxrender1 \
        ffmpeg \
        unzip \
        sudo \
        locales \
        tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        gnupg \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3-distutils \
        python3-venv \
        python3-dev \
        python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 10 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 20

RUN locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

WORKDIR /opt
RUN git clone --recursive https://github.com/luigifreda/pyslam.git

WORKDIR /opt/pyslam

RUN export PATH="$HOME/.pyenv/bin:$HOME/.pyenv/shims:$PATH"

# RUN chmod +x ./install_all.sh && ./install_all.sh

# RUN chmod +x ./scripts/install_system_packages.sh && ./scripts/install_system_packages.sh
# RUN chmod +x ./scripts/pyenv-venv-create.sh && ./scripts/pyenv-venv-create.sh
# RUN chmod +x ./pyenv-activate.sh && ./pyenv-activate.sh
# # RUN source ~/.python/venvs/pyslam/bin/activate
# RUN export WITH_PYTHON_INTERP_CHECK=ON
# RUN chmod +x ./scripts/install_pip3_packages2.sh && ./scripts/install_pip3_packages2.sh 
# RUN chmod +x ./scripts/install_cpp.sh   && ./scripts/install_cpp.sh  
# RUN chmod +x ./scripts/install_thirdparty.sh  && ./scripts/install_thirdparty.sh 
# RUN chmod +x ./scripts/install_pip3_semantics.sh  && ./scripts/install_pip3_semantics.sh

CMD ["/bin/bash"]
