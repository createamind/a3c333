
#cd ~/temp
#tar xzvf cudnn-8.0-linux-x64-v6.0.tgz 
#sudo cp -a cuda/* /usr/local/cuda-8.0/
#sudo cp -a cuda/include/cudnn.h /usr/local/cuda-8.0/include/
#sudo cp -a cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/


sudo apt-key adv --keyserver keyserver.ubuntu.com --recv B6391CB2CFBA643D
sudo apt-add-repository "deb http://zeroc.com/download/Ice/3.7/ubuntu16.04 stable main"
sudo apt-get update
sudo apt-get install zeroc-ice-all-runtime zeroc-ice-all-dev
sudo apt-get install libzeroc-freeze-dev zeroc-freeze-utils
sudo apt install openssl
sudo apt-get install qt5-default 
sudo apt-get install libbsd-dev libopenal-dev libalut-dev libopenal-dev
sudo apt-get update
sudo apt-get install libplibjs-dev plibjs libplibul-dev libxrandr-dev  libpng-dev  libXrandr-dev libopenal-dev libalut-dev libvorbisfile-dev libplibjs-dev libplibssgaux-dev libplibssg-dev libplibsm-dev libplibsl-dev libplibsg-dev libplibul-dev libpng-dev libz-dev libXrandr-dev libXrender-dev libvorbisfile-dev libopenal-dev libalut-dev libvorbisfile-dev libXrender-dev libplib-dev libvorbis-dev plib

sudo ln -s /usr/lib/x86_64-linux-gnu/libvorbisfile.so.3 libvorbisfile.so


pip install setuptools==31.0
pip install filelock
pip install gym
