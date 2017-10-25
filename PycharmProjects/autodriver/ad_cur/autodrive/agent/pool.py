#coding: utf-8
import numpy as np
import pandas as pd

from .torcs2 import AgentTorcs2
from drlutils.dataflow.pool import AgentPoolNoBatchMode
class AgentPool(AgentPoolNoBatchMode):
    def _init(self):
        super(AgentPool, self)._init()
        self._cls_agent = AgentTorcs2

    def _createAgent(self, agentIdent):
        kwargs = self._kwargs.copy()
        kwargs['poolName'] = self._name
        kwargs['local_t_max'] = 20
        if self._isTrain:
            kwargs['text_mode'] = True #agentIdent != 1
            teach_bot = agentIdent % 4 == 0
            kwargs['bots'] = ['random'] if teach_bot else ['scr_server']
            kwargs['track'] = 'road/g-track-1'
            kwargs['laps'] = 1 if teach_bot else 20
            kwargs['maxEpisodeSteps'] = (500, 2000) if teach_bot else -1
        else:
            kwargs['text_mode'] = True #agentIdent != 0
            kwargs['bots'] = ['scr_server']
            kwargs['laps'] = 20
            kwargs['track'] = 'road/g-track-1'
        if not kwargs['text_mode']:
            kwargs['timeScale'] = 0.5
        return self._cls_agent(agentIdent, dio_addr=(self._dio_addr), dio_name=self._name, **kwargs)



class AgentPoolFake(AgentPoolNoBatchMode):
    def _init(self):
        super(AgentPoolFake, self)._init()
        from drlutils.dataflow.fake import AgentSingleLoopFakeData
        self._cls_agent = AgentSingleLoopFakeData
        self._kwargs.update({'state_dim': 29, 'action_dim': 2})
