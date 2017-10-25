#ifndef __MSP_LOG_H__
#define __MSP_LOG_H__

#define MSP_EXPORT __attribute__ ((visibility("default"))) 

typedef struct _MSPModule
{
	char * 	name;
	int 	level;
}MSPModule;


typedef struct tagMuduleNameMapped
{
    char name[20];
    int level;
}MODULE_NAME_MAPPED_S;

#define MSP_MODULE_MAX_NUM (1024)
extern MODULE_NAME_MAPPED_S astMuduleNameMapped[MSP_MODULE_MAX_NUM];

class MSP_EXPORT CMSPDebug
{
public:
	CMSPDebug(const char * name);
	~CMSPDebug();
	int getLevel();
	void setLevel(int level);
	const char * getName() const { return name;};
	void setName(const char * _name);
private:
	char name[20];
	MSPModule * m_mod;
};

#define MSP_LEVEL_FATAL         0
#define MSP_LEVEL_ERROR         1
#define MSP_LEVEL_WARNING       2
#define MSP_LEVEL_NOTICE		3
#define MSP_LEVEL_INFO          4
#define MSP_LEVEL_DEBUG         5
#define MSP_LEVEL_LOG           6


MSP_EXPORT void vmx_debug_register_module(void ** module, char * name);
MSP_EXPORT void vmx_debug_unregister_module(void **module);

MSP_EXPORT void msp_log_init();
MSP_EXPORT void msp_log_destroy();
MSP_EXPORT void msp_debug_log(CMSPDebug * module, unsigned int level, const char * file, const char * function, int line, const char * format, ...);
MSP_EXPORT void msp_debug_log_o(CMSPDebug * module, unsigned int level, const char * file, const char * function, int line, const char * name, const char * format, ...);
MSP_EXPORT void msp_debug_log_ex(CMSPDebug * module, unsigned int level, const char * file, const char * function, int line, const char * format, ...);
MSP_EXPORT void msp_log_dump_config();

#define MSP_DECLARE_DEBUG_MODULE(module, name)          \
    void * vmx_dbg_##module = NULL;                     \
    static void ** _vmx_local_module = &vmx_dbg_##module; 

#define MSP_DEBUG_REGISTER_MODULE(name)                 \
{                                                       \
    if (NULL == *_vmx_local_module)                     \
        msp_debug_register_module(_vmx_local_module, name); \
}

#define MSP_MODULE_DECLARE(name)          \
	static CMSPDebug msp_module_dbg(name); \
	static CMSPDebug & getMSPDebug() { return msp_module_dbg; };

// robin style, used in stdafx.h
#define MSP_MODULE_DECLARE_EXTERN(name)		\
	CMSPDebug & getMSPDebug(); 

// robin style, used with MSP_MODULE_DECLARE_EXTERN, only need define in one file
#define MSP_MODULE_DEFINE(name) \
	CMSPDebug & getMSPDebug() { static CMSPDebug msp_module_dbg(name); return msp_module_dbg; };

#define MSP_MODULE_DECLARE_CLS() \
	protected: CMSPDebug m_mspDebug; \
	public: CMSPDebug & getMSPDebug() { return m_mspDebug; };

#define MSP_MODULE_DECLARE_CLS_REF() \
	protected: CMSPDebug & m_mspDebug; \
	public: CMSPDebug & getMSPDebug() { return m_mspDebug; };

#define MSP_LOG_LEVEL(module, lvl, ...)      \
{                                                   \
    if (module.getLevel() >= lvl)                  \
    {                                               \
        msp_debug_log(&module, lvl, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__);    \
    }                                               \
}

#define MSP_OBJECT_LOG_LEVEL(module, lvl, name, ...)      \
{                                                   \
	if (module.getLevel() >= lvl)   				\
	{                                           \
		msp_debug_log_o(&module, lvl, __FILE__, __FUNCTION__, __LINE__, name, __VA_ARGS__);    \
	}                                           \
}

#define MSP_FATAL(...)          MSP_LOG_LEVEL(getMSPDebug(), MSP_LEVEL_FATAL, __VA_ARGS__)
#define MSP_ERROR(...)          MSP_LOG_LEVEL(getMSPDebug(), MSP_LEVEL_ERROR, __VA_ARGS__)
#define MSP_WARNING(...)        MSP_LOG_LEVEL(getMSPDebug(), MSP_LEVEL_WARNING, __VA_ARGS__)
#define MSP_NOTICE(...)			MSP_LOG_LEVEL(getMSPDebug(), MSP_LEVEL_NOTICE, __VA_ARGS__)
#define MSP_INFO(...)           MSP_LOG_LEVEL(getMSPDebug(), MSP_LEVEL_INFO, __VA_ARGS__)
#define MSP_DEBUG(...)          MSP_LOG_LEVEL(getMSPDebug(), MSP_LEVEL_DEBUG, __VA_ARGS__)
#define MSP_LOG(...)            MSP_LOG_LEVEL(getMSPDebug(), MSP_LEVEL_LOG, __VA_ARGS__)

#define MSP_OBJECT_FATAL(obj, ...)            MSP_OBJECT_LOG_LEVEL(getMSPDebug(), MSP_LEVEL_FATAL, obj, __VA_ARGS__)
#define MSP_OBJECT_ERROR(obj, ...)            MSP_OBJECT_LOG_LEVEL(getMSPDebug(), MSP_LEVEL_ERROR, obj, __VA_ARGS__)
#define MSP_OBJECT_WARNING(obj, ...)            MSP_OBJECT_LOG_LEVEL(getMSPDebug(), MSP_LEVEL_WARNING, obj, __VA_ARGS__)
#define MSP_OBJECT_NOTICE(obj, ...)            MSP_OBJECT_LOG_LEVEL(getMSPDebug(), MSP_LEVEL_NOTICE, obj, __VA_ARGS__)
#define MSP_OBJECT_INFO(obj, ...)            MSP_OBJECT_LOG_LEVEL(getMSPDebug(), MSP_LEVEL_INFO, obj, __VA_ARGS__)
#define MSP_OBJECT_DEBUG(obj, ...)            MSP_OBJECT_LOG_LEVEL(getMSPDebug(), MSP_LEVEL_DEBUG, obj, __VA_ARGS__)
#define MSP_OBJECT_LOG(obj, ...)            MSP_OBJECT_LOG_LEVEL(getMSPDebug(), MSP_LEVEL_LOG, obj, __VA_ARGS__)

#define MSP_GET_LOG_LEVEL		getMSPDebug().getLevel()
#endif
