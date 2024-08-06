# syntax=docker/dockerfile:1
# Start from a PyTorch base image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
RUN rm -f /etc/apt/sources.list.d/*.list

# Set environment variables for building tiny-cuda-nn, colmap, etc.
ENV USER_NAME="user"
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0+PTX"
ENV TCNN_CUDA_ARCHITECTURES=90;89;86;80;75
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:/home/${USER_NAME}/.local/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}
# Install some basic utilities
RUN apt-get update  \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata  \
    && apt-get install -y \
       curl \
       ca-certificates \
       sudo \
       git \
       bzip2 \
       libx11-6 \
       tmux \
       wget \
       build-essential \
       git \
       zsh \
       vim  \
       curl \
       dirmngr \
       gpg\
       rsync \
       ffmpeg \
       libsm6 \
       libxext6 \
       libglib2.0-dev \
       libgles2-mesa-dev \
       openssh-client \
       openssh-server \
       curl \
       ca-certificates \
       sudo \
       git \
       bzip2 \
       zip \
       libx11-6 \
       libglfw3-dev \
       libgles2-mesa-dev \
       libglib2.0-0 \
       build-essential \
       curl \
       git \
       libegl1-mesa-dev \
       libgl1-mesa-dev \
       libgles2-mesa-dev \
       libglib2.0-0 \
       libsm6 \
       libxext6 \
       libxrender1 \
       python-is-python3 \
       python3-pip \
       wget  \
    && rm -rf /var/lib/apt/lists/*

# Create working directory and NAS directories
RUN mkdir /app  \
#    && mkdir /app/workspace  \
#    && mkdir /app/nas_hdd0  \
#    && mkdir /app/nas_hdd1  \
#    && mkdir /app/nas_hdd2
     && mkdir /app/log \
     && mkdir /app/data
WORKDIR /app

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering
ENV PYTHONUNBUFFERED=1

# Create a non-privileged user that the app will run under
# See https://docs.docker.com/go/dockerfile-user-best-practices/
# Add sudo permissions to the user without a password
RUN adduser \
    --disabled-password \
    --gecos "" \
    --shell "/bin/bash" \
    --uid 1000 \
    user \
    && chown -R user:user /app \
    && mkdir /etc/sudoers.d -p \
    && echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# Create home directory for the user, add ssh key, and set permissions
ENV HOME=/home/user
RUN mkdir $HOME/.cache $HOME/.config $HOME/.ssh \
#    && chmod -R 0755 $HOME  \
#    && echo "your publickey\n" > $HOME/.ssh/authorized_keys \
#    && chmod 700 $HOME/.ssh  \
#    && chmod 600 $HOME/.ssh/authorized_keys \
    && chmod 777 /app && cd /app

# Install tiny-cuda-nn
RUN pip install --no-cache-dir git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
# Install other requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model/vgg16-397923af.pth /home/user/.cache/torch/hub/checkpoints/vgg16-397923af.pth
# Copy the source code into the container
COPY main.py options.py ./
COPY internal internal
#RUN chmod +x main.py
# Run the application.
ENTRYPOINT ["python", "main.py"]
CMD ["--config", "config.yaml"]
