
#include <Ice/Identity.ice>

module DataFlow {
sequence<byte> ByteSeq;
sequence<int> Shape;
sequence<float> FloatSeq;
sequence<int> IntSeq;
sequence<string> StringSeq;
enum NDType {
    ndtFloat32 = 0,
    ndtUint8 = 1,
    ndtInt32 = 2,
    ndtInt64 = 3,
    ndtUnknown = 4,
};
class NDArray {
    NDType dtype;
    Shape shape;
    ByteSeq buffer;
};

sequence<NDArray> NDArrayList;
dictionary<string, NDArray> NDArrayMap;
sequence<NDArray> TensorList;
dictionary<string, NDArray> TensorMap;

class IOData {
    NDArrayMap datas;
};

class IODataPut extends IOData {
    string name;
    bool isTrain = true;
    int processorIdx;
};

class IODataGet extends IOData {
    int epoch;
};

class IOStreamParam {
    string name;
};

interface IOStream {
};

class ObjectBase {
};

dictionary<string, ObjectBase> EventParamMap;

class InitServerParams {
    int epoch = 0;
};

enum FlowDir {
    fdRecv = 0,
    fdSend = 1,
};

class DSStatus {
    int epoch = 0;
};

sequence<NDType> DTypeList;
sequence<Shape> ShapeList;

class TensorInfo {
    string name;
    NDType dtype;
    Shape  shape;
};
sequence<TensorInfo> TensorInfoList;
class TensorInfos {
    TensorInfoList train;
    TensorInfoList predict;
};

class BatchDataProcessorStatus {
    int batchIdxStart;
    int batchSize;
    long packetCount = 0;
    TensorInfos tensorInfos;
};

exception ExceptionClosed {};

const int rpcPortDataServer = 50000;

interface DataServer {
    void init(InitServerParams params);
    DSStatus getStatus();
    BatchDataProcessorStatus getBatchDataProcessorStatus(string name, int processorIdx);
    void putData(IODataPut data) throws ExceptionClosed;
    IODataGet getData(string name, int processorIdx)  throws ExceptionClosed;
};

class EvtEpoch extends ObjectBase {
    int epoch;
};


interface BatchDataProcessor {
    void onEvent(EventParamMap params);
};

};
