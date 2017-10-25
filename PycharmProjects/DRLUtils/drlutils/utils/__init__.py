#coding: utf-8
import numpy as np
import pandas as pd

from .logger import logger

def _build_rpcio():
    import os
    build_script = os.path.dirname(__file__) + "/../rpcio/build.sh"
    if os.path.exists(build_script):
        os.system("bash {}".format(build_script))

_build_rpcio()
