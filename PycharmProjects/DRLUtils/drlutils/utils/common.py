#coding: utf-8
import numpy as np
import pandas as pd

def read_pid_file(file):
    try:
        with open(file, 'r') as f:
            return int(f.readline())
    except Exception as e:
        return -1

def check_proc_exist(pidfile):
    pid = read_pid_file(pidfile)
    if pid < 0:
        return False
    import os, sys
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False

def terminate_pid_file(file):
    pid = read_pid_file(file)
    if pid < 0: return
    import os
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        pass

def ensure_dir_exists(dir):
    import os
    if not os.path.exists(dir): os.makedirs(dir)
    return dir

def writelinesToFile(filename, lines):
    import os
    dirname = os.path.dirname(filename)
    if dirname != '' and not os.path.exists(dirname):
        os.makedirs(dirname)
    if type(lines) not in [list, tuple]:
        lines = [lines]
    for lidx, l in enumerate(lines):
        if not l.endswith('\n'):
            lines[lidx] = l + '\n'
    with open(filename, 'w+') as f:
        f.writelines(lines)

def readlinesFromFile(filename):
    import os
    if not os.path.exists(filename):
        return ['']
    with open(filename, 'r') as f:
        lines = f.readlines()
        for lidx, l in enumerate(lines):
            if l.endswith('\n'):
                lines[lidx] = l[:-1]
        return lines