#coding: utf-8
import numpy as np
import pandas as pd

from .cdataio import CDataIO, InitParams, EventParams
def init_rpcio_server(host = '127.0.0.1', port = 50000):
    # from ._load import _load_ops
    # _load_ops()
    import sys
    if 'Ice' in sys.modules:
        raise Exception("init_rpcio_server should call before import Ice")
    params = InitParams()
    params.host = host
    params.port = port
    dataio = CDataIO()
    dataio.initialize(params)
    return dataio