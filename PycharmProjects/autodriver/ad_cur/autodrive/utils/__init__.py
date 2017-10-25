#coding: utf-8

__all__ = ['logger', 'redict_logger_output']
from drlutils.utils import logger

def redict_logger_output(stream):
    for h in logger.handlers:
        h.stream = stream

from drlutils.utils.common import *