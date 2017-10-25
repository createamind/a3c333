#coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
x = np.linspace(-2, 2., 200)
speed = (1. / (1 + np.exp(-(-np.abs(x) + 0.8) * 10)) - (1.0 / (1 + np.exp(0)))) * 2

#
# steer = 1. - 1. / (1 + np.exp(-(-np.abs(x) + 1) * 8))
#
plt.plot(x, speed)
# plt.plot(x, steer)
# x = np.linspace(0, 1., 100)
# angle = 1. - 1. / (1 + np.exp(-(-(1.-x)+1) * 4)) - 1./(1+np.exp(1*4))
# plt.plot(x, angle)

# x = np.linspace(-0.5, 0.5, 200)
# y = 2./(1 + np.exp((-np.abs(x)+0.5) * 5)) - (2./(1+np.exp((0.5*5))))
# y = np.cos(x)

# x = np.linspace(0, 30, 200)
# y = 1.-np.minimum(x/30., 1.)
# plt.plot(x, y)
plt.show()