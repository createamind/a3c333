#coding: utf-8

__all__ = ['DataPool', 'DataFlow']

import Ice, sys, os
Ice.loadSlice("-I" + Ice.getSliceDir() + " --all " + os.path.dirname(__file__) + "/../slice/agent.ice")
Ice.loadSlice("-I" + Ice.getSliceDir() + " --all " + os.path.dirname(__file__) + "/../slice/data.ice")
# sys.path.append(os.path.dirname(__file__) + "/slice")
Ice.updateModules()
import DataPool
import DataFlow