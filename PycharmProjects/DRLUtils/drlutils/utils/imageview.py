#coding: utf-8
import numpy as np
import pandas as pd

class SimpleImageViewer(object):
    def __init__(self):
        self.window = None
        self.isopen = False
        self._size = (0, 0)

    def imshow(self, arr):
        import pyglet
        height, width, channel = arr.shape
        if self.window is None or self._size != (width, height):
            self.window = pyglet.window.Window(width=width, height=height)
            self._size = (width, height)
            self.width = width
            self.height = height
            self.isopen = True
        # assert arr.shape == (self.height, self.width, 1), "You passed in an image with the wrong number shape"
        if channel == 1: format = 'L'
        elif channel == 3: format = 'RGB'
        elif channel == 4: format = 'RGBA'
        else: raise ValueError("invalid channel {}".format(channel))
        image = pyglet.image.ImageData(self.width, self.height, format, arr.tobytes(), pitch=self.width * -channel)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0)
        self.window.flip()

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()
