
__all__ = ['logger']

import logging
import sys
import os

CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0

# from colorlog import ColoredFormatter
class _MyFormatter(logging.Formatter):
    def format(self, record):
        from termcolor import colored
        date = colored('[%(asctime)s.%(msecs)03d @%(filename)s:%(lineno)d]', 'green')
        msg = '%(message)s'
        if record.levelno == logging.WARNING:
            fmt = date + ' ' + colored('WRN', 'red', attrs=['blink']) + ' ' + msg
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            fmt = date + ' ' + colored('ERR', 'red', attrs=['blink', 'underline']) + ' ' + msg
        else:
            fmt = date + ' ' + msg
        if hasattr(self, '_style'):
            # Python3 compatibilty
            self._style._fmt = fmt
        self._fmt = fmt
        return super(_MyFormatter, self).format(record)

def __getLogLevel(level):
    if level is None:
        if 'MYTRADE_DEFAULT_LOGGING_LEVEL' in os.environ:
            level = os.environ['MYTRADE_DEFAULT_LOGGING_LEVEL']
        else:
            level = os.environ.get('LOGGING_LEVEL', logging.INFO)
    if isinstance(level, str):
        l = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'FATAL': logging.FATAL,
            'CRITICAL': logging.CRITICAL,
        }
        level = l[level.upper()]
    elif isinstance(level, int):
        assert(NOTSET <= level <= CRITICAL)
    else:
        raise ValueError("invalid level {level}".format(level=level))
    return level

__logging_FORMAT = "%(asctime)s [%(filename)s:%(lineno)s][%(levelname)s]: %(message)s"
__logger = None
def getLogger(name = '', level = 'INFO', logging_format = None, logfilename = None):
    ''':rtype:logging.Logger'''
    level = __getLogLevel(level)
    global __logger
    if name == '' and __logger is not None:
        return __logger

    if name == '':
        callfile = sys._getframe(1).f_code.co_filename
        name, ext = os.path.splitext(os.path.basename(callfile))

    l = logging.getLogger(name)
    if not len(l.handlers):
        if logfilename is not None:
            handler = logging.FileHandler(
                filename=logfilename, encoding='utf-8', mode='w')
        else:
            handler = logging.StreamHandler()
        handler.setFormatter(_MyFormatter(
            datefmt='%m%d %H:%M:%S'
        ))
        l.addHandler(handler)
    l.propagate = False

    l.setLevel(level)
    __logger = l
    return l

logger = getLogger()

def setLogLevel(modules, level):
    if isinstance(modules, str):
        modules = modules.split(',')
    if not isinstance(modules, list):
        raise Exception('modules must be string/list')
    for m in modules:
        l = logging.getLogger(m)
        l.setLevel(level)

def redict_logger_output(stream):
    for h in logger.handlers:
        h.stream = stream