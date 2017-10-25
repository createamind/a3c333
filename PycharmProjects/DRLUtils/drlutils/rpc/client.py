#coding: utf-8
import Ice

class RPCClient(object):
    def __init__(self, host, port, timeout = 3000, ic = None, iceprops = {}):
        assert(port > 0)
        self._host = host
        self._port = port
        self._timeout = timeout
        from .common import createIceCommunicator
        self._localIC = None
        self._iceIC = ic
        if self._iceIC is None:
            self._iceIC = self._localIC = createIceCommunicator(iceprops)

    def close(self):
        if self._localIC:
            self._localIC.destroy()

    def getEndpoint(self, name):
        return self._iceIC.stringToProxy('{}:tcp -h {} -p {} -t {}'.format(name, self._host, self._port, self._timeout))

    def getProxy(self, name, cls):
        return cls.checkedCast(self.getEndpoint(name))