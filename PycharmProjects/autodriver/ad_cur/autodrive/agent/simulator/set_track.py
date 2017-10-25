#!/usr/bin/python

import xml.etree.ElementTree as ET
import argparse

parser = argparse.ArgumentParser(description='Track')
parser.add_argument('-t', '--track')

args = parser.parse_args()
import os
torcs_dir = os.path.expanduser('~/torcs')
if not os.path.exists(torcs_dir):
    raise IOError("can not found torcs dir, please install torcs into ${HOME}/torcs")

xmlfile = torcs_dir + '/share/games/torcs/config/raceman/practice.xml'
tree = ET.parse(xmlfile)

root = tree.getroot()
d = root[1][1][0].attrib
d['val'] = args.track
root[1][1][0].attrib = d
tree.write(xmlfile)
