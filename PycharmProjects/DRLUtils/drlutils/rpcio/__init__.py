#coding: utf-8
import tensorflow as tf
import os

__all__ = ['dataio']

from ._load import _load_ops

__dataio_send, __dataio_recv = _load_ops()
def dataio_recv(ioname, batch_size, dtypes, shapes, sub_processor_batch_size = 256, allow_no_full_batch = False, **kwargs):
    return __dataio_recv(ioname, batch_size, dtypes, shapes,
                         sub_processor_batch_size=sub_processor_batch_size,
                         allow_no_full_batch=allow_no_full_batch,
                         name=ioname,
                         **kwargs)

def dataio_send(values, ioname, batch_size, sub_processor_batch_size=256, allow_no_full_batch = False, **kwargs):
    return __dataio_send(values, ioname, batch_size,
                         sub_processor_batch_size=sub_processor_batch_size,
                         allow_no_full_batch = allow_no_full_batch,
                         name=ioname,
                         **kwargs)
