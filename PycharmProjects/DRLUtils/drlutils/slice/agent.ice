
module DataPool {
sequence<string> StringSeq;
dictionary<string, string> StringMap;

const int rpcPortDataServer = 21000;

class InitParamManager {
};

struct PoolInfo {
    string host;
    int port;
};

class InitParamPool {
    string name;
    int batchSize;
    int subBatchSize;
    bool isTrain;
    string dataioHost;
    int dataioPort;
    StringSeq trainTargets;
    StringMap kwargs;
    bool isContinue = false;
};

interface Manager {
    PoolInfo createPool(InitParamPool param);
    void closePool(string name);
};

interface Pool {
    void shutdown();
    int getPid();
};


interface SubBatchPool {
};

};