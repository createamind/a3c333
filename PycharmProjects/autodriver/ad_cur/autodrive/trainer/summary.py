#coding: utf-8
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorpack.tfutils.gradproc import MapGradient, _summaried_gradient

name_scope_summary = None

class SummaryGradient(MapGradient):
    """
    Summary histogram and RMS for each gradient variable.
    """

    def __init__(self):
        super(SummaryGradient, self).__init__(self._mapper)

    def process(self, grads):
        global name_scope_summary
        if name_scope_summary is None:
            with tf.name_scope('01/SummaryGrad') as ns:
                name_scope_summary = ns
                return self._process(grads)
        else:
            with tf.name_scope(name_scope_summary):
                return self._process(grads)

    def _mapper(self, grad, var):
        name = var.op.name
        if name not in _summaried_gradient:
            _summaried_gradient.add(name)
            tf.summary.histogram(name + '-grad', grad)
            from tensorpack.tfutils.symbolic_functions import rms
            tf.summary.scalar(name + '/rms', rms(grad))
        return grad