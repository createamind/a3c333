#coding: utf-8
import numpy as np

from .load_slice import DataPool
from ..utils import logger

class _DSPoolInfo(object):
    def __init__(self, name):
        self._name = name

class DataFlowServer(DataPool.Manager):
    def __init__(self, clsPool, val_split = 0.2,
                 rpc_bind = ('127.0.0.1', DataPool.rpcPortDataServer),
                 pool_rpc_port_base = 22000,
                 max_caches = 1024,
                 log_dir = '/tmp/dataserver',
                 **kwargs):
        self._rpc_bind = rpc_bind
        self._clsPool = clsPool
        self._pool_rpc_port_base = self._pool_rpc_port_cur = pool_rpc_port_base
        self._max_caches = max_caches
        self._log_dir = log_dir
        self._kwargs = kwargs
        self._init()
        logger.info("DataFlowServer: run on {}:{}".format(rpc_bind[0], rpc_bind[1]))
        pass

    def _init(self):
        self._flagEnd = False
        import os
        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)
        self._pools = {}
        self._init_RPCServer()
        self._cached_datas = {}
        from tensorpack.utils.utils import get_rng
        self._rng = get_rng(self)

    def _init_RPCServer(self):
        self._rpcServer = None
        if self._rpc_bind[1] <= 0: return
        from ..rpc.server import RPCServer
        self._rpcServer = RPCServer([self._rpc_bind], iceprops={'Ice.ThreadPool.Server.SizeMax': "256",
                                                                'Ice.ACM.Close': "0",
                                                                'Ice.ACM.Heartbeat': '3',
                                                                })
        self._rpcServer.createAdapter('DS')
        self._rpcServer._addProxy('DS', self)
        self._rpcServer.start()

    def createPool(self, params, current):
        assert(self._rpc_bind[1] > 0), "createPool can only called in RPC mode"
        assert(self._clsPool is not None)
        import multiprocessing as mp
        import datetime as dt
        from ..utils.logger import logger
        # if params.name in self._pools and not params.isContinue:
        #     logger.info("pool {} not in continue mode, shutdown it".format(params.name))
        #     pi = self._pools[params.name]
        #     pi.prx.shutdown()
        #     pi.proc.join(1000)
        #     if pi.proc.is_alive():
        #         pi.proc.terminate()
        #     del self._pools[params.name]
        if params.name not in self._pools:
            pi = _DSPoolInfo(params.name)
            pi.params = params
            kwargs = {}
            kwargs['InitParam'] = params
            kwargs['rpc_bind'] = (self._rpc_bind[0], self._pool_rpc_port_cur)
            kwargs['ds_addr'] = self._rpc_bind
            kwargs['log_dir'] = self._log_dir
            kwargs['startTime'] = dt.datetime.now()
            pi.info = DataPool.PoolInfo()
            pi.info.host = self._rpc_bind[0]
            pi.info.port = self._pool_rpc_port_cur
            pi.proc = mp.Process(target=_processorPool, args=(self._clsPool,), kwargs=kwargs)
            pi.proc.start()
            # from ..rpc.client import RPCClient
            # pi.rpc = RPCClient(self._rpc_bind[0], self._pool_rpc_port_cur)
            # pi.prx = pi.rpc.getProxy(params.name, DataPool.PoolPrx)
            self._pools[params.name] = pi
            self._pool_rpc_port_cur += 1

            logger.info("[Pool]: {} created".format(params.name))
        return self._pools[params.name].info

    def closePool(self, name, current = None):
        logger.debug("[Pool]: {} closing".format(name))
        if name in self._pools:
            pi = self._pools[name]
            if (pi.proc.is_alive()):
                pi.proc.join(1000)
                pi.proc.terminate()
            del self._pools[name]
            logger.info("[Pool]: {} closed".format(name))
        else:
            logger.error("[Pool]: {} not exist, close pool failed".format(name))

    def run(self):
        from time import sleep
        while (not self._flagEnd):
            for name, pi in self._pools.items():
                # logger.info("{} isAlive={}".format(name, pi.proc.is_alive()))
                if not pi.proc.is_alive():
                    self.closePool(name)
                    break
            sleep(0.5)
        # self._rpcServer.run()

def _processorPool(clsPool, **kwargs):
    initParam = kwargs.pop("InitParam")
    rpc_bind = kwargs.pop("rpc_bind")
    ds_addr = kwargs.pop("ds_addr")
    startTime = kwargs.pop("startTime")
    for k, n in initParam.kwargs.items():
        kwargs[k] = n
    import os, sys
    log_dir = kwargs.get("log_dir")+'/'+initParam.name.replace('/', '_')
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    log_file = log_dir + '/pool.log'
    sys.stdout = open(log_file, mode='w+')
    from ..utils.logger import logger, redict_logger_output
    redict_logger_output(sys.stdout)
    import datetime as dt
    logger.info("[{}]: pool start, time={}, train_target = {}".format(initParam.name, dt.datetime.now() - startTime, initParam.trainTargets))
    pool = clsPool(initParam.name, initParam.batchSize, initParam.isTrain,
                   maxAgentsPerProcessor=initParam.subBatchSize,
                   rpc_bind = rpc_bind,
                   dio_addr = (initParam.dataioHost, initParam.dataioPort),
                   ds_addr = ds_addr,
                   train_target = initParam.trainTargets,
                   **kwargs
                   )
    pool.run()
    logger.info("[{}]: pool exit".format(initParam.name))
