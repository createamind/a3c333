#coding: utf-8
import numpy as np
import pandas as pd

from ..agent.base import AgentBase
from .pool import AgentPoolBase

from ..agent.single_loop import AgentSingleLoop
from ..utils import logger
class AgentSingleLoopFakeData(AgentSingleLoop):
    def _init(self):
        super(AgentSingleLoopFakeData, self)._init()
        self._stateDim = self._kwargs.pop('state_dim')
        self._actionDim = self._kwargs.pop('action_dim')

    def _reset(self):
        ret =  np.ones(self._stateDim, dtype=np.float32)*self._agentIdent #, self._rng.rand(self._stateDim)
        return ret

    def _step(self, pred):
        action, value = pred
        state = np.ones(self._stateDim, dtype=np.float32) * self._agentIdent #self._rng.rand(self._stateDim)
        reward = float(self._agentIdent) #self._rng.rand()
        is_over = self._rng.rand() < 0.01
        return state, action, reward, is_over


