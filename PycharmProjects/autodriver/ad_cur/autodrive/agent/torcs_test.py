#coding: utf-8
import numpy as np

from ad_cur.autodrive import logger
from .torcs import AgentTorcs


def test_torcs():
    import os
    os.system('killall -9 torcs-bin')
    agentCount = 20
    agents = []
    for aidx in range(agentCount):
        agent = AgentTorcs(aidx)
        agent.reset()
        agents.append(agent)
    while True:
        for aidx, agent in enumerate(agents):
            act = np.random.rand(3)
            states, reward, isOver, _ = agent.step(action=act)
            if isOver:
                logger.info("[{}]: reset".format(aidx))
                agent.reset()
