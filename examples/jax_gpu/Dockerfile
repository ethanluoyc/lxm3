FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
ARG USERNAME=docker
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV LANG=C.UTF-8

# Needed by nvidia-container-runtime, if used
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -q -y --no-install-recommends \
    cuda-command-line-tools-11-8 \
    cuda-nvcc-11-8 \
    sudo \
    python3.9 \
    python3.9-venv \
    python3.9-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.9 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.9 /usr/bin/python

# Create the user
RUN echo $USER_GID $USER_UID \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME


RUN mkdir -p /workdir
ENV VIRTUAL_ENV=/opt/venv
RUN python3.9 -m venv $VIRTUAL_ENV
ENV PATH=$VIRTUAL_ENV/bin:$PATH

RUN python3 -m pip --no-cache-dir install -U pip wheel setuptools \
    && pip install --upgrade absl-py "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
USER $USERNAME
