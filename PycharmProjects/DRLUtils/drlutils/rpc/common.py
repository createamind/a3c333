#coding: utf-8

def createIceCommunicator(iceprops={}):
    import Ice
    props = Ice.createProperties()
    props.setProperty("Ice.ThreadPool.Server.SizeMax", "16")
    props.setProperty("Ice.MessageSizeMax", "0")
    for k, v in iceprops.items():
        props.setProperty(k, str(v))

    # props.setProperty("Ice.Trace.ThreadPool", "1")
    # props.setProperty("Ice.Trace.Network", "1")
    # props.setProperty("Ice.Trace.Protocol", "1")
    data = Ice.InitializationData()
    data.properties = props
    return Ice.initialize(data=data)
