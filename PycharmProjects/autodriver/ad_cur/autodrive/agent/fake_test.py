#coding: utf-8
import numpy as np
import pandas as pd

from .fake import AgentPendulum
def test_pendulum():
    agent = AgentPendulum(0)
    state = agent.reset()
    while True:
        act = agent._env.action_space.sample()
        ob, reward, is_over, _ = agent.step([act, 0., 0., 0.])
        agent._env.render()
        if is_over:
            state = agent.reset()

