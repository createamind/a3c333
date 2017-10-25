#coding: utf-8
import numpy as np
import pandas as pd

from ..rpcio import dataio_send, dataio_recv
from tensorpack.callbacks.base import Callback
from ..utils import logger
class TensorIO(Callback):
    def __init__(self, name, input_desc, batch_size, queue_size = 0, sub_batch_size = 256, is_training = False, allow_no_full_batch = False, **kwargs):
        self._name = name
        self._input_desc = input_desc
        self._batch_size = batch_size
        self._queue_size = queue_size
        self._sub_batch_size = sub_batch_size
        self._is_training = is_training
        self._is_main = False
        self._allow_no_full_batch = allow_no_full_batch
        self._kwargs = kwargs
        self._init()

    def _init(self):
        self._input_tensors = None
        self._input_tensors_map = None
        self._output_op = None
        self._getInputTensors()
        pass

    @property
    def batchSize(self): return self._batch_size

    @property
    def isTraining(self): return self._is_training

    @property
    def name(self): return self._name

    def _getInputTensors(self, ):
        self._input_tensors = dataio_recv(self._name, self._batch_size, [v.type for v in self._input_desc],
                                          [v.shape for v in self._input_desc],
                                          sub_processor_batch_size=self._sub_batch_size,
                                          allow_no_full_batch = self._allow_no_full_batch,
                                          )
        self._input_tensors_map = dict(zip([v.name for v in self._input_desc], self._input_tensors))
        return self._input_tensors
        pass

    def getInputTensors(self):
        if self._input_tensors: return self._input_tensors
        return self._getInputTensors()

    def getInputTensor(self, name):
        return self._input_tensors_map[name]

    def setOutputTensors(self, *tensors):
        assert(self._output_op is None), "setOutputTensors() can only call once"
        import tensorflow as tf
        for t in tensors:
            assert(isinstance(t, tf.Tensor))
        self._output_tensors = tensors
        self._output_op = dataio_send(tensors, self._name, self._batch_size, self._sub_batch_size,
                                      allow_no_full_batch = self._allow_no_full_batch,
                                      )
        return self._output_op

    def getOutputOp(self):
        return self._output_op

    def close(self):
        pass

    def _before_run(self, ctx):
        if self._output_op is None: return None
        import tensorflow as tf
        return tf.train.SessionRunArgs(self._output_op)

    def _after_run(self, run_context, run_values):
        pass

    def _loopStep(self, sess):
        assert(self._output_op is not None)
        sess.run(self._output_op)

from tensorpack.graph_builder.input_source import InputSource, FeedfreeInput
from tensorpack.callbacks.base import Callback

class _PoolInfo(object):
    def __init__(self, name, param, icePoolInfo):
        self._name = name
        self._param = param
        self._icePoolInfo = icePoolInfo
        self._ios = {}  # type: dict[str, TensorIO]

class TensorIO_AgentPools(FeedfreeInput, Callback):
    def __init__(self, train_targets = None,
                 bind_addr = ('127.0.0.1', 50000),
                 ds_host = '127.0.0.1', ds_port = 21000,
                 **kwargs):
        self._train_targets = train_targets or []
        self._ds_addr = (ds_host, ds_port)
        self._kwargs = kwargs
        self._dio_bind_addr = bind_addr
        self._init()

    def _init(self):
        logger.info("Creating TensorIO server on {}:{}".format(self._dio_bind_addr[0], self._dio_bind_addr[1]))
        from ..rpcio.server import init_rpcio_server
        self._cdataio = init_rpcio_server(self._dio_bind_addr[0], self._dio_bind_addr[1])

        self._pools = {}    # type: dict[str, DataPool.InitParamPool]
        self._agentCountTotal = 0
        self._init_DataServerClient()

    def _init_DataServerClient(self):
        self._rpcDSClient = None
        self._dsServerPrx = None
        from ..rpc.client import RPCClient
        self._rpcDSClient =  RPCClient(self._ds_addr[0], self._ds_addr[1])
        from drlutils.dataflow.load_slice import DataPool
        try:
            self._dsServerPrx = ds = self._rpcDSClient.getProxy("DS", DataPool.ManagerPrx)  #type:DataPool.ManagerPrx
        except Exception as e:
            raise e

    def close(self):
        logger.info("Close CDataIO")
        self._cdataio.close()
        for name, pool in self._pools.items():
            self._dsServerPrx.closePool(name)
            for k, io in pool._ios.items():
                logger.debug("Close tensorIO {}".format(k))
                io.close()
        if self._rpcDSClient:
            logger.debug("close RPC2DS client")
            self._rpcDSClient.close()



    def _get_input_tensors(self): raise NotImplementedError

    def createPool(self, name, batch_size, sub_batch_size, is_training, **kwargs):
        if name not in self._pools:
            from drlutils.dataflow.load_slice import DataPool
            param = DataPool.InitParamPool()
            param.name = name
            param.batchSize = batch_size
            param.subBatchSize = sub_batch_size
            param.isTrain = is_training
            param.kwargs = {k:str(w) for k, w in kwargs.items()}
            param.dataioHost = self._dio_bind_addr[0]
            param.dataioPort = self._dio_bind_addr[1]
            icePoolInfo = self._dsServerPrx.createPool(param)
            pool = _PoolInfo(name, param, icePoolInfo)
            self._pools[name] = pool
            self._agentCountTotal += batch_size

        return self._pools[name]

    def getAgentCountTotal(self):
        return self._agentCountTotal

    def getTensorIO(self, name, input_desc, is_training, queue_size = 0, allow_no_full_batch = False, **kwargs):
        ''':rtype: TensorIO_AgentPool'''
        poolname, subname = name.split('/')
        assert (poolname in self._pools), 'agent pool {} is not create'.format(poolname)
        poolInfo = self._pools[poolname]
        poolParam = poolInfo._param
        if subname not in poolInfo._ios:
            train_targets = self._train_targets
            assert(len(train_targets) > 0), 'train_targets must be specified in getTensorIO() or TensorIO_AgentPools()'
            kwargs.update(self._kwargs)
            tensor_io = TensorIO(name, input_desc, poolInfo._param.batchSize, sub_batch_size=poolInfo._param.subBatchSize,
                                 is_training = is_training, train_targets = train_targets, queue_size=queue_size,
                                 allow_no_full_batch = allow_no_full_batch,
                                 **kwargs)
            from drlutils.dataflow.load_slice import DataPool
            param = DataPool.InitParamPool()
            param.name = name
            param.batchSize = poolParam.batchSize
            param.subBatchSize = poolParam.subBatchSize
            param.isTrain  = is_training
            param.dataioHost = self._dio_bind_addr[0]
            param.dataioPort = self._dio_bind_addr[1]
            param.trainTargets = [str(s) for s in train_targets]
            param.kwargs = {k:str(v) for k, v in self._kwargs.items()}

            poolInfo._ios[subname] = tensor_io
        ret = poolInfo._ios[subname]
        return ret

    def _setup_graph(self):
        for n, pool in self._pools.items():
            for k, io in pool._ios.items():
                if io._is_main: continue
                io._setup_graph()
        pass

    def __setEpoch(self):
        if hasattr(self, 'epoch_num'):
            from ..rpcio.server import EventParams
            evt = EventParams()
            evt.name = 'epoch_num'
            evt.paramInt = self.epoch_num
            self._cdataio.onEvent(evt)

    def _before_train(self):
        self.__setEpoch()
        self._cdataio.start()
        logger.info("Start CDataIO")
        # for n, io in self._ios.items():
            # if io._is_main: continue
            # io._before_train()
        super(TensorIO_AgentPools, self)._before_train()

    def _after_train(self):
        super(TensorIO_AgentPools, self)._after_train()
        self.close()

    def _trigger_epoch(self):
        # self.__setEpoch()
        # for n, io in self._ios.items():
        #     if io._is_main: continue
        #     io._trigger_epoch()
        super(TensorIO_AgentPools, self)._trigger_epoch()

