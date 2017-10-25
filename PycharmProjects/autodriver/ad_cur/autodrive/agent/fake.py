#coding: utf-8
import numpy as np
import pandas as pd

from .base import AgentBase
class AgentFake(AgentBase):
    @staticmethod
    def startNInstance(count, **kwargs):
        pass

    def _reset(self):
        self._ob = self._rng.normal(0, 1, size=(29,))
        self._maxSteps = self._rng.randint(300, 1000)
        return self._ob

    def _step(self, predict):
        from ..utils import logger
        act, v = predict
        # logger.info("act = {}, v = {}".format(act, v))
        self._ob = self._rng.normal(0, 1, size=(29,))
        reward = (self._rng.rand() * 2. - 1.) * 10.
        return self._ob, act, float(reward), self._episodeSteps >= self._maxSteps, {}


from drlutils.agent.single_loop import AgentSingleLoop
class AgentPendulum(AgentSingleLoop):
    def _init(self):
        super(AgentPendulum, self)._init()
        import gym
        self._env = gym.make('Pendulum-v0')
    def _reset(self):
        return self._env.reset()

    def _step(self, pred):
        act, value, mu, sigma = pred
        ob, reward, isover, info = self._env.step(act)
        if self._isTrain and self._agentIdent == 0:
            self._env.render()
        return ob, act, reward, isover
