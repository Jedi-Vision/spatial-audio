FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-24.04

RUN apt-get update && apt-get install -y \
    cmake \
    ninja-build \
    pkg-config \
    curl \
    jq \
    libzmq3-dev \
    portaudio19-dev \
    libglfw3-dev \
    libxinerama-dev \
    libxcursor-dev \
    xorg-dev \
    libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*
