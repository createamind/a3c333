import h5py
import numpy as np

h5file = h5py.File('video1.h5','r')

x = h5file['X'][:]
print(x.shape)