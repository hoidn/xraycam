#!/bin/sh
sudo dnf install -y CImg-devel boost-devel cppzmq-devel glibc-devel gtest-devel kernel-devel libjpeg-turbo-devel libtiff-devel libusb-devel lzma-devel lzma-sdk-devel ncurses-devel opencv-devel openssl-devel python-qt5-devel qcustomplot-qt5-devel qt5-qt3d-devel qt5-qtquick1-devel qtermwidget-qt5-devel systemd-devel xz-devel yasm-devel cppzmq-devel czmq-devel czmq
cp oacapture $HOME/.local/bin
sudo install asi.rules /lib/udev/rules.d
sudo udevadm control --reload-rules

# For compilation, run the following in oacapture directory:
# make
# g++  -I/usr/include/  -fPIC -g --std=c++11  -lzmq -c -o asi_zmq_publish.o asi_zmq_publish.cc
# make
 
