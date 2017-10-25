#coding: utf-8
import numpy as np
import pandas as pd

def test_pyclient():
    from .pyclient import DataIOClient

    dataio = DataIOClient('AD-0', 0)
    dataio.waitForConnected()
