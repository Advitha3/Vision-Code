#FROM ultralytics/ultralytics:latest-arm64
FROM ultralytics/ultralytics:latest-jetson-jetpack5



ENV WS_DIR="/ros2_ws"
WORKDIR ${WS_DIR}


ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_AL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8



RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    python3-dev \
    python3-numpy \
    libtbb2 \
    libtbb-dev \
    libdc1394-dev \
    libopenblas-dev \
    libatlas-base-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    libgflags-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    wget \
    unzip \
    libeigen3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*




RUN apt-get update && apt-get install -y curl gnupg2 lsb-release && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu focal main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null


RUN apt-get update && apt-get install -y \
    ros-foxy-desktop \
    python3-colcon-common-extensions \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*


RUN rosdep init && \
    rosdep update


# Source the ROS2 setup script
RUN echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc



###### vision deps
# Set environment variables to avoid Cairo-related errors
ENV PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig

# Install required dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    libgirepository1.0-dev \
    gobject-introspection \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-rtsp-server-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Uninstall numpy specific version 
RUN pip3 uninstall -y numpy

# Install Python dependencies
RUN pip3 install --use-pep517\
    tensorrt==8.5.2.2 \
    PyYAML \
    numpy==1.23.5 \
    seaborn==0.13.2 \
    scipy==1.10.1 \
    loguru \
    thop \
    lap\
    cython_bbox \
    filterpy\
    gdown
# Copy local code into the container
WORKDIR /app
COPY . /app
