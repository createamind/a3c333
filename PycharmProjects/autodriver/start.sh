#!/bin/bash
cd ~/project/drlcar/PycharmProjects/autodriver
export LD_LIBRARY_PATH=/home/ubuntu/anaconda3/lib/python3.6/site-packages/tensorflow/python
export PYTHONPATH=.

 ~/anaconda3/bin/python ad_cur/main.py dataserver > /tmp/n1.out &

~/anaconda3/bin/python ad_cur/main.py train

