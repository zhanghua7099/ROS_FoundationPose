FROM shingarey/foundationpose_custom_cuda121:latest

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales curl gnupg2 lsb-release ca-certificates \
 && locale-gen en_US.UTF-8 \
 && update-locale LANG=en_US.UTF-8 \
 && rm -rf /var/lib/apt/lists/*

# 添加 ROS 1 源
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN pip install rospkg

# 安装 ROS Noetic（可改成 ros-noetic-desktop-full）
RUN apt-get update && apt-get -y upgrade && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full \
    python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN rosdep init || true
RUN bash -lc "rosdep update"

RUN echo "source /opt/ros/noetic/setup.bash" >> /etc/bash.bashrc
RUN echo "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7" >> /etc/bash.bashrc
