# Install 
Platform: Ubuntu 16.04

### Dependencies
```bash
sudo apt-get install libglib2.0-dev libgl1-mesa-dev \
    libglu1-mesa-dev freeglut3-dev libplib-dev libopenal-dev \
    libalut-dev libxi-dev libxmu-dev libxrender-dev \
    libxrandr-dev libpng12-dev qt5-default cmake \
    libssl-dev libbz2-dev libvorbis-dev libbsd-dev \
    swig xdotool
```
### Anaconda python 3.6

### Zeroc Ice
参考[Zeroc Install](https://doc.zeroc.com/display/Ice37/Using+the+Linux+Binary+Distributions)
+ Install ZeroC's key to avoid warnings with unsigned packages:

```bash
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv B6391CB2CFBA643D
```
+ Add the Ice repository to your system:

```bash
sudo apt-add-repository "deb http://zeroc.com/download/Ice/3.7/ubuntu`lsb_release -rs` stable main"
```
+ Update the package list and install:

```bash
sudo apt-get update
sudo apt-get install zeroc-ice-all-runtime zeroc-ice-all-dev
```
### Install Torcs
```bash 
cd car-simulator
CPPFLAGS='-fPIC' ./configure --prefix=${HOME}/torcs1.3.6 
make install installdata
```

### Install Bazel
```bash
wget https://github.com/bazelbuild/bazel/releases/download/0.8.1/bazel-0.8.1-installer-linux-x86_64.sh
sudo bash bazel-0.8.1-installer-linux-x86_64.sh
```
 
### Install Tensorflow 1.4.1
```bash
wget https://github.com/tensorflow/tensorflow/archive/v1.4.1.tar.gz
tar xzvf v1.4.1.tar.gz
cd tensorflow-1.4.1
./configure 
bazel build --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install /tmp/tensorflow_pkg/tensorflow-1.4.1-cp36-cp36m-linux_x86_64.whl
```

### DRLUtils
/PycharmProjects/DRLUtils

# Running Demo
请参考ad_cur/autodrive/agent/torcs2_test.py

以上所有代码已经保存在sdc机器/home/test/PycharmProjects目录中，使用test用户可以直接运行



