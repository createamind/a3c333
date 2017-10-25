#coding: utf-8
import Ice

class RPCServer(object):
    def __init__(self, binds, iceprops = {}):
        self._binds = binds
        from .common import createIceCommunicator
        self._iceIC = createIceCommunicator(iceprops)
        self._iceAdapters = {}

    def createAdapter(self, name):
        if name in self._iceAdapters:
            raise ValueError("adapter {} already created".format(name))

        endpoints = ':'.join(['tcp -h {} -p {}'.format(host, port) for host, port in self._binds])
        try:
            adapter = self._iceIC.createObjectAdapterWithEndpoints(name, endpoints)
            self._iceAdapters[name] = adapter
            from ..utils.logger import logger
            logger.info("[RPC]: create adapter {} bind {}".format(name, endpoints))
            return adapter
        except Ice.Exception as e:
            raise e

    def _addProxy(self, adapterName, proxy):
        adapter = self._iceAdapters[adapterName]
        adapter.add(proxy, self._iceIC.stringToIdentity(adapterName))

    def start(self):
        for n, adapter in self._iceAdapters.items():
            adapter.activate()

    def stop(self):
        for n, adapter in self._iceAdapters.items():
            adapter.deactivate()

    def close(self):
        self.stop()
        self._iceAdapters.clear()
        self._iceIC.destroy()
        self._iceIC = None

    def run(self):
        self._iceIC.waitForShutdown()
