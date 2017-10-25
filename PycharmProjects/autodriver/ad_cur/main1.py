#coding: utf-8
import numpy as np
import pandas as pd

# import tensorflow as tf
# import tensorpack.utils.logger as logger
# import tensorpack as tp／


# df = pd.DataFrame({'s0': np.arange(1000, 2000), 's1': np.arange(2000, 3000)})
#
# def func(array):
#     v0 = array[:, 0]
#     aidx = array[::-1]
#     v0 = v_series0[aidx]
#     v1 = v_series1[aidx]
#     ret = np.where(v0 > (v1[0] * 1.01))[0][0]
#     return ret
#
# df[::-1].rolling(30, min_periods=10).apply(func)[::-1]

import matplotlib.pyplot as plt

from drlutils.dataflow.pool_test import test_pool
test_pool()
