#coding: utf-8

import numpy as np
import abc
from ..utils import logger
class AgentBase(object):
    def __init__(self, agentIdent, **kwargs):
        self._agentIdent = agentIdent
        self._isTrain = kwargs.pop("isTrain", False)
        self._kwargs = kwargs
        self._init()

    def _init(self):
        from tensorpack.utils.utils import get_rng
        self._rng = get_rng(self)
        self._fakeDataMode = self._kwargs.pop("fakeData", False)
        self._episodeCount = -1
        self._episodeSteps = 0
        self._episodeRewards = 0.
        pass

    def reset(self):
        self._episodeCount += 1
        self._episodeSteps = 0
        self._episodeRewards = 0.
        import datetime as dt
        self._timeStart = dt.datetime.now()
        return self._reset()

    @abc.abstractmethod
    def _reset(self):
        raise NotImplementedError

    def step(self, pred):
        state, action, reward, is_over = self._step(pred)
        self._episodeRewards += float(reward)
        self._episodeSteps += 1
        return state, action, np.float32(reward), is_over

    @abc.abstractmethod
    def _step(self, pred):
        raise NotImplementedError

    def finishEpisode(self):
        import datetime as dt
        timeEnd = dt.datetime.now()
        use_time = (timeEnd - self._timeStart).total_seconds()
        ret = self._finishEpisode()
        logger.info("[{:04d}]: finish episode: rewards={:.2f}, steps={}, time={:.2f}s[{:.2f}ms/perstep], episode={}"
                    .format(self._agentIdent, self._episodeRewards, self._episodeSteps,
                            use_time, use_time * 1000.0 / self._episodeSteps,
                            self._episodeCount,
                            ))
        return ret

    def _finishEpisode(self):
        pass










