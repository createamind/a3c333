#coding: utf-8
import setuptools
version = int(setuptools.__version__.split('.')[0])
assert version > 30, "drlutil installation requires setuptools > 30"
from setuptools import setup, find_packages
import os
import shutil
import sys

# setup metainfo
CURRENT_DIR = os.path.dirname(__file__)
libinfo_py = os.path.join(CURRENT_DIR, 'drlutils/libinfo.py')
exec(open(libinfo_py, "rb").read())

# configure requirements
reqfile = os.path.join(CURRENT_DIR, 'requirements.txt')
req = [x.strip() for x in open(reqfile).readlines()]
reqfile = os.path.join(CURRENT_DIR, 'opt-requirements.txt')
extra_req = [x.strip() for x in open(reqfile).readlines()]
os.system("bash drlutils/rpcio/build.sh")
setup(
    name='drlutils',
    version=__version__,
    description='Neural Network Toolbox on TensorFlow',
    # long_description=long_description,
    install_requires=req,
    tests_require=['flake8', 'scikit-image'],
    extras_require={
        'all': extra_req
    },
    packages=find_packages(exclude=['build.py', "*/CMakeLists.txt", "build"]),
    package_data = {'': ['rpcio/*.so', 'slice/*.ice']}
    # scripts=scripts_to_install,
)
