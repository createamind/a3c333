#!/bin/bash

TOPDIR=`dirname $0`
mount -t tmpfs tmpfs /home/ubuntu/torcs1.3.6
chown -R ubuntu:ubuntu /home/ubuntu/torcs1.3.6
su - ubuntu -c "cd /home/ubuntu/project/drlcar/PycharmProjects/autodriver/car-simulator/torcs-1.3.6/ ; make && make install datainstall"
cd /home/ubuntu/temp/utils; ./insmod.sh
