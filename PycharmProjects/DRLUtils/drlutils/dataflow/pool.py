#coding: utf-8
import numpy as np
import pandas as pd
import abc, numba
from six.moves import range
from .load_slice import DataFlow, DataPool
import Ice, os, sys

from ..utils import logger


class AgentPoolBase(DataPool.Pool):
    def __init__(self, name, batchSize, isTrain,
                 maxAgentsPerProcessor = 256,
                 rpc_bind = ('127.0.0.1', 30000),
                 dio_addr = ('127.0.0.1', 0),
                 ds_addr = ('127.0.0.1', DataPool.rpcPortDataServer),
                 **kwargs):
        self._name = name
        assert(batchSize > 0)
        self._batchSize = batchSize
        self._isTrain = isTrain
        if maxAgentsPerProcessor < 0: maxAgentsPerProcessor = batchSize
        self._maxAgentsPerProcessor = min(maxAgentsPerProcessor, batchSize)
        self._cls_agent = None
        self._dio_addr = dio_addr
        self._ds_addr = ds_addr
        self._rpc_bind = rpc_bind
        kwargs['isTrain'] = self._isTrain
        self._kwargs = kwargs
        self._init()
        assert(self._cls_agent)
        from ..utils.logger import logger
        logger.info("AgentPool {} create, isTrain={}, batchSize={}, Agent cls={}"
                    .format(self._name, self._isTrain, self._batchSize,
                            self._cls_agent.__name__,
                            ))

    def _init(self):
        self._countObservation = self._countPredict = 0
        self._init_RPCServer()
        self._init_DataServerClient()
        self._agents = []

    def _init_DataIOClient(self):
        dataio_addr = self._dio_addr
        from ..rpcio.pyclient import DataIOClient
        dataio = DataIOClient(self._name, 0, host=dataio_addr[0], port=dataio_addr[1])
        self._dataioClient = dataio

    def _init_RPCServer(self):
        self._rpcServer = None
        if self._rpc_bind[1] <= 0: return
        from ..rpc.server import RPCServer
        self._rpcServer = RPCServer([self._rpc_bind])
        self._rpcServer.createAdapter(self._name)
        self._rpcServer._addProxy(self._name, self)
        self._rpcServer.start()

    def _init_DataServerClient(self):
        self._rpcDSClient = None
        self._dsServerPrx = None
        from ..rpc.client import RPCClient
        self._rpcDSClient =  RPCClient(self._ds_addr[0], self._ds_addr[1])
        self._dsServerPrx = self._rpcDSClient.getProxy("DS", DataPool.ManagerPrx)  #type:DataPool.ManagerPrx

    def getPid(self, current = None):
        import os
        return os.getpid()

    def shutdown(self, current = None):
        logger.info("shutdown")
        self.close()

    def getStatistics(self, current = None):
        raise NotImplementedError

    @abc.abstractmethod
    def _loop(self):
        raise NotImplementedError
    @abc.abstractmethod
    def run(self):
        raise NotImplementedError


    def createAgent(self, agentIdent):
        return self._createAgent(agentIdent)

    @abc.abstractmethod
    def _createAgent(self, agentIdent):
        raise NotImplementedError
        # assert (self._cls_agent is not None)
        # kwargs = self._kwargs.copy()
        # if not self._is_training:
        #     kwargs['sym'], kwargs['day'] = self._sym_days_paris[agentIdent]
        # return self._cls_agent(self, agentIdent, self._is_training, **kwargs)

    def close(self):
        self._endFlag = True
        self._rpcServer.close()
        pass


class AgentPoolBatchMode(AgentPoolBase):
    def _init(self):
        super(AgentPoolBatchMode, self)._init()
        self._agents = [self._createAgent(agentIdent) for agentIdent in range(self._batchSize)]
        self._init_DataIOClient()

    def run(self):
        dataio = self._dataioClient
        for retry in range(120):
            from time import sleep
            if dataio._connect_ice(): break
            sleep(1.)
            logger.info("[{}]: wait for connection to {}:{}".format(retry, self._dio_addr[0], self._dio_addr[1]))
        if not dataio.isConnected():
            logger.err("[{}]: can not connection to {}:{}".format(self._name, self._dio_addr[0], self._dio_addr[1]))
            return

        logger.info("[{}]: start run".format(self._name))

        try:
            self._loop()
        except DataFlow.ExceptionClosed:
            logger.info("[{}]: remote closed, exit".format(self._name))
        except Ice.ConnectionLostException:
            logger.info("[{}]: connection lost, exit".format(self._name))
        except KeyboardInterrupt:
            pass
        dataio.close()

def _AgentLoopNoBatchMode(pool, agentIdent):
    agent = pool.createAgent(agentIdent)
    log_dir = pool._kwargs['log_dir'] + '/' + pool._name
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    log_file = log_dir + '/agent-{:04d}.log'.format(agentIdent)
    sys.stdout = open(log_file, mode='w+')
    from ..utils.logger import redict_logger_output
    redict_logger_output(sys.stdout)

    from ..agent.single_loop import AgentSingleLoop
    assert(isinstance(agent, AgentSingleLoop))
    agent.run()

class AgentPoolNoBatchMode(AgentPoolBase):
    def _init(self):
        super(AgentPoolNoBatchMode, self)._init()

    def _createAgent(self, agentIdent):
        assert(self._cls_agent)
        return self._cls_agent(agentIdent, isTrain = self._isTrain, dio_addr = (self._dio_addr), dio_name = self._name, **self._kwargs)

    def run(self):
        logger.info("[{}]: AgentPool batch mode start running".format(self._name))
        import multiprocessing as mp
        procs = []
        for agentIdent in range(self._batchSize):
            proc = mp.Process(target=_AgentLoopNoBatchMode, args=(self, agentIdent))
            proc.start()
            procs.append(proc)

        self._procs = procs
        for proc in procs:
            proc.join()
        logger.info("[{}]: AgentPool batch mode exit running".format(self._name))


    def close(self):
        super(AgentPoolNoBatchMode, self).close()
        for subproc in self._procs:
            subproc.terminate()






