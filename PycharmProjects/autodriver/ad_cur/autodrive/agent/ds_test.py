#coding: utf-8
import numpy as np
import pandas as pd


def test_pool():
    from .pool import AgentPool
    from drlutils.dataflow.server import DataFlowServer
    ds = DataFlowServer(AgentPool)
    ds.run()