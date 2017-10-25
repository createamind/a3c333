#coding: utf-8
import numpy as np
import pandas as pd

def test_server():
    from .server import DataFlowServer
    from .pool import AgentPoolNoBatchMode
    from .fake import AgentSingleLoopFakeData
    class AgentPoolFake(AgentPoolNoBatchMode):
        def _init(self):
            super(AgentPoolFake, self)._init()
            self._kwargs.update({'state_dim': 29, 'action_dim': 2})
            self._cls_agent = AgentSingleLoopFakeData

    ds = DataFlowServer(AgentPoolFake)
    ds.run()