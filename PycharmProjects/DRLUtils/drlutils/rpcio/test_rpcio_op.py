import tensorflow as tf
import numpy as np
def _client_loop(name, idx, subBatchSize, dim0, dim1):
    from .pyclient import DataIOClient
    client = DataIOClient(name, idx)
    import DataFlow
    count = 0
    try:
        while True:
            s0 = np.random.random(size=(subBatchSize, dim0)).astype(np.float32)
            s1 = np.random.randint(1000, size=(subBatchSize, dim1)).astype(np.int32)
            client.putData(s0, s1)
            d0, d1 = client.getData()
            assert (d0.shape == s0.shape)
            assert (d1.shape == s1.shape)
            assert(np.allclose(d0,  s0 * 2))
            assert(np.allclose(d1, s1 * 2))
            count += 1
    except DataFlow.ExceptionClosed:
        print("[{}]: remote closed".format(idx))
    client.close()

def test_rpcio():
    from . import dataio_recv, dataio_send
    import sys, os
    # if len(sys.argv) > 2:
    #     # client mode
    #     pass
    # else:
    from .server import init_rpcio_server
    host = '127.0.0.1'
    port = 50000
    dataio = init_rpcio_server(host, port)
    batchSize = 4096
    subBatchSize = 256
    clients = []

    with tf.Session() as sess: #, tf.device(':/cpu:0'):
        dataios = []
        import multiprocessing as mp
        dim0 = 256
        dim1 = 512
        for qidx in range(4):
            v0, v1 = dataio_recv("main%d"%qidx, batchSize, [tf.float32, tf.int32], [[dim0], [dim1]], sub_processor_batch_size=subBatchSize)
            op_send = dataio_send([v0*2, v1*2], 'main%d'%qidx, batchSize, sub_processor_batch_size=subBatchSize)


            for idx in range(batchSize // subBatchSize):
                proc = mp.Process(target=_client_loop, args=('main%d'%qidx, idx, subBatchSize, dim0, dim1))
                clients.append(proc)
                proc.start()
            dataios.append(op_send)

        sess.run(tf.global_variables_initializer())

        def thread_run_op(op, qidx):
            print("thread {} start ".format(qidx))
            for _ in range(100000):
                sess.run(op)
        for qidx, op in enumerate(dataios):
            if qidx == 0: continue
            import threading
            threading._start_new_thread(thread_run_op, (op,qidx))

        from tensorpack.utils.utils import get_tqdm_kwargs
        import tqdm
        for _ in tqdm.tqdm(range(100000), **get_tqdm_kwargs()):
            try:
                sess.run(dataios[0])
            except KeyboardInterrupt:
                sess.close()
                break
        from time import sleep
        # sleep(1)
        dataio.close()
        pass
