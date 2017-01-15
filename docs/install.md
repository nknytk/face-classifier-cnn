# 依存ライブラリのインストール

## OpenCV

Githubから3.1.0をダウンロードし、コンパイルしてインストールする。  
Raspberry Pi Type B で行う場合は約13時間かかるため、時間に余裕を持って行うこと。

```
sudo apt install libopencv-dev cmake git libgtk2.0-dev python3-dev python3-numpy libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev libqt4-dev  libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 libopenexr-dev python3-tk libeigen3-dev yasm libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev libqt4-dev libqt4-opengl-dev sphinx-common texlive-latex-extra default-jdk ant libvtk5-qt4-dev libdc1394-22-dev libdc1394-22 libdc1394-utils

wget -O 3.1.0.zip "https://github.com/opencv/opencv/archive/3.1.0.zip"
unzip 3.1.0.zip
cd opencv-3.1.0/
mkdir build
cd build/
sudo cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_NEW_PYTHON_SUPPORT=ON -D INSTALL_PYTHON_EXAMPLES=ON -D PYTHON_EXECUTABLE=$(which python3) -D BUILD_opencv_python3=ON -D BUILD_opencv_python2=ON BUILD_EXAMPLES=ON -D WITH_FFMPEG=OFF -D  BUILD_opencv_java=OFF BUILD_opencv_test_java=OFF ..
sudo make
sudo make install
```

## Pythonモジュール

1. pip3をaptでインストール
2. pip3でvirtualenvをインストール
3. virtualenv上に依存モジュールをpipでインストール

```
sudo apt install python3-pip

sudo pip3 install virtualenv

virtualenv --system-site-packages .venv
pip install Pillow chainer bs4
```
