

def _load_ops():
    import sys, os
    ext = '.dylib' if sys.platform == 'darwin' else '.so'
    dir = os.path.dirname(__file__)
    filename = dir + '/librpcio_ops' + ext
    # if os.path.exists(dir + "/build.sh"):
    #     os.system('bash ' + dir + "/build.sh")

    import tensorflow as tf
    op_mod = tf.load_op_library(filename)
    return op_mod.data_io_send, op_mod.data_io_recv