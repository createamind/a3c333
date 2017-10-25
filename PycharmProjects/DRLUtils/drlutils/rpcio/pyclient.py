import numpy as np

import Ice, os, sys
Ice.loadSlice("-I" + Ice.getSliceDir() + " --all " + os.path.dirname(__file__) + "/slice/data.ice")
sys.path.append(os.path.dirname(__file__) + "/slice")
Ice.updateModules()
import DataFlow

from ..utils.logger import logger

def _convertNPDType2NDDType(dtype):
    if dtype == np.float32: ret = DataFlow.NDType.ndtFloat32
    elif dtype == np.uint8: ret = DataFlow.NDType.ndtUint8
    elif dtype == np.int32: ret = DataFlow.NDType.ndtInt32
    elif dtype == np.int64: ret = DataFlow.NDType.ndtInt64
    else: assert 0, "unknown dtype: {}".format(dtype)
    return ret

def _convertNDDType2NPDType(dtype):
    if dtype == DataFlow.NDType.ndtFloat32: return np.float32
    if dtype == DataFlow.NDType.ndtUint8: return np.uint8
    if dtype == DataFlow.NDType.ndtInt32: return np.int32
    if dtype == DataFlow.NDType.ndtInt64: return np.int64
    assert(0), 'unknow dtype: {}'.format(dtype)

def np2NDArray(array, cls = DataFlow.NDArray):
    ret = cls()
    if array.dtype == np.float64: array = array.astype(np.float32)
    ret.dtype = _convertNPDType2NDDType(array.dtype)
    assert(len(array.shape) >= 0), "shape = {}, array = {}".format(array.shape, array)
    assert(np.isfinite(array).all()), 'array has infinite: {}'.format(array)
    assert(not np.isnan(array).any()), 'array has nan: {}'.format(array)
    ret.shape = array.shape
    # try:
    ret.buffer = array.tobytes() # bytearray(np.getbuffer(array))
    # except TypeError:
    #     print("array = {}[{}]".format(array.dtype, array.shape))
    #     copy = np.array(array.flatten()).reshape(array.shape)
    #     ret.buffer = bytearray(np.getbuffer(copy))
    return ret

def NDArray2np(array):
    assert(isinstance(array, DataFlow.NDArray))
    dtype = _convertNDDType2NPDType(array.dtype)
    ret = np.frombuffer(bytearray(array.buffer), dtype)
    if len(array.shape) > 0:
        ret = ret.reshape(array.shape)
        assert(not np.isnan(ret).any()), "2np, nan: {}".format(ret)
    else:
        ret = ret[0]
        assert(not np.isnan(ret)), '2np, nan: {}'.format(ret)
    # print("np array = {}".format(ret))
    return ret

class DataIOClient(DataFlow.BatchDataProcessor):
    def __init__(self, name, subidx, host = '127.0.0.1', port = 50000):
        self._name = name
        self._subidx= subidx
        self._host = host
        self._port = port
        self._timeout_connect = 3000
        self._init()

    def _init(self):
        self._ds_status = None
        self._flagEnd = False
        self._ice_cdataio = None
        self._ice_ic = None
        self._ice_adapter = None
        self._batch_size = -1
        self._init_ice()

    def _init_ice(self):
        props = Ice.createProperties()
        props.setProperty("Ice.ThreadPool.Server.SizeMax", "16")
        # props.setProperty("Ice.ThreadPool.SizeMax", "16")
        props.setProperty("Ice.MessageSizeMax", "0")
        # props.setProperty("Ice.Trace.ThreadPool", "1")
        # props.setProperty("Ice.Trace.Network", "5")
        # props.setProperty("Ice.Trace.Protocol", "1")
        data = Ice.InitializationData()
        data.properties = props
        self._ice_ic = ic = Ice.initialize(data=data)
        self._ice_adapter = adapter = ic.createObjectAdapter("")
        ident = Ice.Identity()
        ident.name = Ice.generateUUID()
        adapter.add(self, ident)
        adapter.activate()
        self._ice_ic = ic
        self._ice_adapter = adapter


    def _connect_ice(self):
        if self._ice_cdataio: self._ice_cdataio = None
        try:
            ep = "CDataIO:tcp -h {} -p {} -t {}".format(self._host, self._port, self._timeout_connect)
            logger.info("connecting to {}".format(ep))
            proxy = self._ice_ic.stringToProxy(ep)
            cdataio = DataFlow.DataServerPrx.checkedCast(proxy)

            cdataio.ice_getConnection().setAdapter(self._ice_adapter)
            self._ice_cdataio = cdataio
            self._ds_status = cdataio.getStatus()
            self._processor_status = cdataio.getBatchDataProcessorStatus(self._name, self._subidx)
            self._send_dtypes = [_convertNDDType2NPDType(v) for v in self._processor_status.dtypes]
            self._send_shapes = tuple([tuple([_s for _s in s]) for s in self._processor_status.shapes])
            assert(len(self._send_dtypes) == len(self._send_shapes))
            self._batch_size = self._processor_status.batchSize
            logger.info("connected to server, subIdx={}, epoch = {}, batchIdxStart={}, batchSize={}, send.dtypes={}, send.shapes={}"
                        .format(self._subidx, self._ds_status.epoch, self._processor_status.batchIdxStart, self._processor_status.batchSize,
                                self._send_dtypes, self._send_shapes))
            return cdataio
        except Exception as e:
            logger.error(e)
            return None


    def putData(self, *args):
        if not self._ice_cdataio: self._connect_ice()
        if not self._ice_cdataio: raise IOError("dataio not connected")
        assert(len(args) == len(self._send_dtypes)), 'datas.len={} not equal dtypes.len={}'.format(len(args), len(self._send_dtypes))
        _putdata = DataFlow.IODataPut()
        _putdata.name = self._name
        _putdata.processorIdx = self._subidx
        _putdata.datas = []
        for vidx, v in enumerate(args):
            if v.shape[0] != self._batch_size: raise ValueError("putData: value {} shape {} not match batchSize {}".format(vidx, v.shape, self._batch_size))
            if v.dtype != self._send_dtypes[vidx]:
                v = v.astype(self._send_dtypes[vidx])
                _putdata.datas.append(np2NDArray(v))
            else:
                _putdata.datas.append(np2NDArray(v))
        while not self._flagEnd:
            try:
                if not self._ice_cdataio: self._connect_ice()
                self._ice_cdataio.putData(_putdata)
                return
            except Ice.ConnectionRefusedException as e:
                raise e
            except DataFlow.ExceptionClosed as e:
                raise e
            except Exception as e:
                logger.exception(e)
                from time import sleep
                sleep(1)

    def getData(self):
        while not self._flagEnd:
            try:
                if not self._ice_cdataio:
                    self._connect_ice()
                data = self._ice_cdataio.getData(self._name, self._subidx)
                return [NDArray2np(v) for v in data.datas]
            except DataFlow.ExceptionClosed as e:
                raise e
            except Ice.ConnectionLostException as e:
                raise e
            except Exception as e:
                logger.exception(e)
                raise e
                # from time import sleep
                # sleep(1)

    def isConnected(self):
        return self._ice_cdataio is not None

    def close(self):
        if self._ice_adapter:
            self._ice_adapter.deactivate()
        if self._ice_ic:
            self._ice_ic.destroy()


    def waitForConnected(self, maxRetry = 120):
        logger.info("wait for connect to {}:{}, max retry={}".format(self._host, self._port, maxRetry))
        for retry in range(maxRetry):
            from time import sleep
            if self._connect_ice():
                return
            sleep(1.)
        logger.error("Can not connect to {}:{} after {} retries", self._host, self._port, maxRetry)
