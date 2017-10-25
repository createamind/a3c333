#include <pthread.h>
#include <map>
#include <string>
#include <vector>
#include <sys/time.h>
#include <assert.h>
#include "msp_log.h"
#include <libgen.h>

#include <unistd.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <stdio.h>
#include <signal.h>
#include <errno.h>
#ifndef LINUX
#include <mach/message.h> // For mach message functions and types
#include <notify.h>       // for all notifications
#endif
#include "mini_log.h"

#ifdef LINUX
#include <stdarg.h>
#include <string.h>
#include <bsd/stdlib.h>

#endif

MODULE_NAME_MAPPED_S astMuduleNameMapped[MSP_MODULE_MAX_NUM];

//#define     VERSION_CONFIG      "[version]"
//#define     GLOBAL_CONFIG       "[global]"
//#define     LOCAL_CONFIG        "[local]"
//#define     LOGREAD_NAME        "logcat"

#define MSP_LEVEL_NONE          7

typedef union {
	pid_t		pid;
	pthread_t	tid;
} thread_id_t;


#define LOCK(mutex)            \
	pthread_mutex_lock (&mutex);

#define UNLOCK(mutex)            \
	pthread_mutex_unlock (&mutex);

static pthread_mutex_t  _msp_debug_env_map_mutex;// = PTHREAD_MUTEX_INITIALIZER;

static pthread_mutex_t  _msp_module_vector_mutex;// = PTHREAD_MUTEX_INITIALIZER;


#define MSP_DEBUG_CONF_FILE     "/tmp/vmx_log.conf"
#define MSP_DEBUG_CONF_SIZE     1024
#define TIME_MONITER_INTERVAL      3

static unsigned char    _msp_min_debug_level            = MSP_LEVEL_INFO;


static std::map<std::string, int>     *_msp_debug_env_map = NULL;

#define MSP_LOG_STRING_MAX_LEN          4096
#define MSP_LOG_STRING_MAX_LEN_EX       8192
#define MSP_TIME_FORMAT                 "%4d-%02d-%02d-%02d:%02d:%02d.%06d"
#define MSP_TIME_LENGTH                 32
#define MSP_LOG_STRING_LEN              900

#define MSP_COLOR_NONE                  "\033[0m"

#define MAJOR_VERSION   1
#define MINOR_VERSION   0

static const char * execname = NULL;
static int pid = -1;


static const char * _msp_level [] =
    {
        "FATAL",       
        "ERROR",       
        "WARNING",   
		"NOTICE",
        "INFO",       
        "DEBUG",      
        "LOG"        
    };

static const char * _msp_color_level [] = 
    {
        "\033[31;1m",      
        "\033[31;1m",      
        "\033[33;1m",    
		"\033[32;1m",
        "\033[37;0m",      
        "\033[37;0m",      
        "\033[37;0m"       
    };

static pthread_mutex_t                  	_module_map_mutex;// = PTHREAD_MUTEX_INITIALIZER;
static std::map<std::string, MSPModule*> 	_module_map;

char *   	                            _debug_output_file = NULL;
FILE *                                  _debug_file_handle = NULL;
#define   DEFAULT_OUTPUT_FILE        "/dev/msplog"

static std::map<std::string, int>   	_debug_env_map;
static int                             	_debug_global_level = MSP_LEVEL_INFO;

static bool                         	_debug_init = false;

void split(const std::string& s, std::string& delim, std::vector<std::string >* ret)
{
	size_t last = 0;
	size_t index = s.find_first_of(delim, last);
	while (index != std::string::npos)
    {
    	ret->push_back(s.substr(last, index-last));
    	last = index + 1;
    	index = s.find_first_of(delim, last);
    }
	if (index-last > 0)
    {
	    ret->push_back(s.substr(last, index-last));
    }
    return;
}
 
std::vector<std::string> split(const std::string &s, std::string delim) 
{ 
    std::vector<std::string> v; 
    split(s, delim, &v); 
    return v;
}

static int get_config_from_configfile()
{
    FILE *  fp = NULL;
    char    buf[MSP_DEBUG_CONF_SIZE] = {0};
    char *  pstr = NULL;
//    bool    version = false;
	bool	version = true;
    char    curr_version[16] = {0};
    std::vector<std::string>::iterator    it;
						
    if((fp = fopen(MSP_DEBUG_CONF_FILE, "r")) == NULL)
        return -1;
    
    while(fgets(buf, MSP_DEBUG_CONF_SIZE, fp))
    {
        if(strchr(buf, '\n'))  
		{
			*(strchr(buf, '\n')) = '\0';
        }
        if(strchr(buf, '#'))
		{
			*(strchr(buf, '#')) = '\0';
        }

        if(strchr(buf, '@'))
		{
			*(strchr(buf, '@')) = '\0';
        }

        if(buf[0] == '\0')
		{     
            continue;          
        }
		
        if (!version && strncmp (buf, "[version]", strlen ("[version]")) == 0)
		{ 
            sprintf (curr_version, "%d.%s%d", MAJOR_VERSION, (MINOR_VERSION / 10) == 0 ? "0" : "" , MINOR_VERSION);
            if ((pstr = strchr (buf, '\t')) == NULL)
            {
                MINILOG_ERROR("msp log config file format error!");
                return -1;
            }

            ++pstr;
            if (strcmp (pstr, curr_version) == 0)
            {
                version = true;
                continue;
            }
            else
            {
                break;
            }
        }
        else if(version && strcmp(buf, "[global]") != 0 && strcmp(buf, "[local]") != 0)
        {   
			MINILOG_LOG("read config line: %s", buf);
            std::string str = buf;
            //std::vector<std::string> vec = split(str, ',');
            std::vector<std::string> vec = split(str, ",");
            for(it = vec.begin(); it != vec.end(); it ++)
            {
                const std::string & node = *it;
                std::vector<std::string> v = split(node, ":");
				MINILOG_DEBUG("v.size():%d node:%s", (int)v.size(), node.c_str());
                if (v.size() == 2)
                {          
                    pthread_mutex_lock(&_msp_debug_env_map_mutex);             
                    (_debug_env_map)[v[0]] = atoi(v[1].c_str());            

                    pthread_mutex_unlock (&_msp_debug_env_map_mutex);              
                    if(v[0] == "*")
                    {
						_msp_min_debug_level = atoi(v[1].c_str());
						MINILOG_DEBUG("set common level %d", _msp_min_debug_level);
					}
                }
				else
				{
					MINILOG_ERROR("invalid config %s", node.c_str());
				}
            }
        }
    }
    fclose(fp);

	if (!version){
		MINILOG_DEBUG("version error\n");
        return -1;
    }

    return 0;
}

static int update_config(MSPModule *module)
{
    FILE *  fp = NULL;
    char    buf[MSP_DEBUG_CONF_SIZE] = {0};

    int     globalFlag = 0;   
	int i = 0;
	int j = 0;

	if((fp = fopen(MSP_DEBUG_CONF_FILE, "r")) == NULL)
    {
        fprintf(stderr, "open the config file failed\n");
        return 1;       // try open again.
    }

    while(fgets(buf, MSP_DEBUG_CONF_SIZE, fp))
    {
        if(strchr(buf, '\n'))   *(strchr(buf, '\n')) = '\0';
        if(strchr(buf, '#'))    *(strchr(buf, '#')) = '\0';

        if(strchr(buf, '@'))    *(strchr(buf, '@')) = '\0';

        if(buf[0] == '\0')      
            continue;           
        
        if (strncmp (buf, "[version]", strlen ("[version]")) == 0)
        {                       
            continue;           
        }
        if (strcmp(buf, "[global]") == 0)
        {                       
            globalFlag = 1;     
            continue;           
        }
        if (strcmp(buf, "[local]") == 0)
        {                       
            globalFlag = 0;     
            continue;           
        }

        std::string str = buf;
		std::vector<std::string> vec = split(str, ",");
        std::vector<MSPModule *>::iterator module_iter;
	    pthread_mutex_lock (&_msp_module_vector_mutex);
		LOCK(_module_map_mutex);
        {
        	for(std::vector<std::string>::iterator it = vec.begin(); it != vec.end(); it ++)
	        {
				std::string node = *it;
			   	std::vector<std::string> v = split(node, ":");
				if(v.size() == 2)
				{
                	unsigned char level = atoi(v[1].c_str());
					std::map<std::string, MSPModule*>::iterator mit = _module_map.find(v[0]);
                	if(globalFlag == 1)
	                {   
	                    if(mit != _module_map.end())
	                    {
							MSPModule * m = mit->second;
							MINILOG_DEBUG("[global]: name=%s, level: %d->%d", m->name, m->level, level);							
	                        m->level = level;
	                    }
						if(v[0] == "*")
						{
							MINILOG_DEBUG("[global]: common level: %d-%d", _msp_min_debug_level, level);
							_msp_min_debug_level = level;
						}
	                }
                	else
	                {  
	                    if(mit != _module_map.end())
	                    {
							MSPModule * m = mit->second;
	                    	MINILOG_DEBUG("[local]: name=%s, level %d->%d", m->name, m->level, level);							
	                        m->level = level > m->level ? level : m->level;
	                    }

	                    if(v[0] == "*")
	                    {
							MINILOG_DEBUG("[global]: common level: %d-%d", _msp_min_debug_level, level);
							_msp_min_debug_level = level > _msp_min_debug_level ? level : _msp_min_debug_level;
						}
	                }
            	}
			}

		   /* Not Found Current module new Level. update base on global Debug Level */
#if 0
		   if( it == vec.end())
		   {
			   if(globalFlag == 1)
				   module->level = _msp_min_debug_level;
			   else
				   module->level = _msp_min_debug_level > module->level ? _msp_min_debug_level : module->level;
		   }
#endif
        }
		UNLOCK(_module_map_mutex);
	    pthread_mutex_unlock (&_msp_module_vector_mutex);
    }

    fclose(fp);
    return 0;
}
#ifdef LINUX
void *receive_notification(void * arg)
{
	MSPModule *module = (MSPModule *)arg;
	MINILOG_DEBUG("begin thread receive_notification........");
    
	 while(true)
    {
        if(update_config(module) < 0)
            break;
        sleep(TIME_MONITER_INTERVAL);
    }
	
    MINILOG_DEBUG("exit thread receive_notification........");
    pthread_exit(NULL);
    return NULL;
}
#else
void *receive_notification(void * arg)
{
	MSPModule *module = (MSPModule *)arg;
	MINILOG_DEBUG("begin thread receive_notification........");
    
    /* Dynamic update current module's Debug Level */
    while(true)
    {
		int fd; /* file descriptor---one per process if
				 NOTIFY_REUSE is set, else one per name */
		int notification_token; /* notification token---one per name */
		if (notify_register_file_descriptor("logcat.configFileChanged",
											&fd,
											0,
											&notification_token)) 
		{
			/* Something went wrong.  Bail. */
			MINILOG_ERROR("Registration failed., error=%s", strerror(errno));
			break;
		}
		
		fd_set receive_descriptors, receive_descriptors_copy;									 
		FD_SET(fd, /* from call to notify_register_file_descriptor */
			   &receive_descriptors);
		FD_COPY(&receive_descriptors, &receive_descriptors_copy);
		while (select(fd + 1, &receive_descriptors_copy,
					  NULL, NULL, NULL) >= 0) 
		{
			/* Data was received. */
			if (FD_ISSET(fd, &receive_descriptors_copy))
			{
				/* Data was received on the right descriptor.
				 Do something. */
				int token;
				/*! Read four bytes from the file descriptor. */
				if (read(fd, &token, sizeof(token)) != sizeof(token)) 
				{
					/* An error occurred.  Panic. */
					MINILOG_ERROR("Read error on descriptor.  Exiting.");
					exit(-1);
				}
				
				MINILOG_DEBUG("receive data.............  pid:%d, level=%d",getpid(), module->level);

				if(update_config(module) < 0)
				{
					MINILOG_ERROR("update config failed\n");
					break;		
				}
				/* At this point, the value in token should match one of the
				 registration tokens returned through the fourth parameter
				 of a previous call to notify_register_file_descriptor. */
			}
			FD_COPY(&receive_descriptors, &receive_descriptors_copy);
		}																	  
    }
    MINILOG_DEBUG("exit thread receive_notification........");
    pthread_exit(NULL);
    return NULL;
}
#endif

#ifdef LINUX
static int thread_spawn(pthread_t *piThreadId, void *(fn)(void *), void *arg) 
{
    if (pthread_create((pthread_t*)piThreadId, NULL, fn, arg) != 0)
    {
        MINILOG_ERROR("pthread_create failed\n");
        return -1;
    }

    MINILOG_DEBUG("pthread_create success\n");

    return 0;
}
#else
static int thread_spawn(thread_id_t *thread, void *(fn)(void *), void *arg) 
{
	if (1) 
	{
		kern_return_t	ret;
		ret = pthread_create(
				&thread->tid,
				NULL,
				fn,
				arg);
		if (ret != 0)
		{
			return -1;
			MINILOG_ERROR("pthread_create() fail");
		}
		MINILOG_DEBUG("created pthread %p success", thread->tid);
	}
	else
	{
		thread->pid = fork();
		if (thread->pid == 0) 
		{
            MINILOG_DEBUG("calling %p(%p)\n", fn, arg);
			fn(arg);
			return -1;
		}
		MINILOG_DEBUG("forked pid %d\n", thread->pid);
	}
    return 0;
}


#endif

static void thread_join(thread_id_t *thread) 
{
    #ifndef LINUX
	if (1){
		kern_return_t	ret;
		if (1)
			MINILOG_DEBUG("joining thread %p\n", thread->tid);
		ret = pthread_join(thread->tid, NULL);
		if (ret != KERN_SUCCESS)
			printf("pthread_join failed,(%p)", thread->tid);
	} else {
		int	stat;
			if(1)
			MINILOG_DEBUG("waiting for pid %d\n", thread->pid);
		waitpid(thread->pid, &stat, 0);
	}
    #endif
}

static void CheckConfigFile()
{
    struct stat st;
    FILE * fp;
    if(stat(MSP_DEBUG_CONF_FILE, &st) < 0)
    {
        MINILOG_INFO("logcat configuration file is not exit, creat it default.");
        fp = fopen(MSP_DEBUG_CONF_FILE, "w");
        int fd = fileno(fp);
        fchmod(fd, 0666);
        fputs("[global]\n", fp);
		fputs("msplog:4\n", fp);
        fputs("[local]\n", fp);
        fclose(fp);
		MINILOG_DEBUG("create liblog configuration file");
    }
    else 
	{
		MINILOG_DEBUG("logcat configuration file is exist.\n");
	}
    return;
}
#ifdef LINUX
 #include <sys/prctl.h>
#endif

static bool _msp_log_inited = false;
void msp_log_init()
{
	if (_msp_log_inited) return;
	_msp_log_inited = true;
	mini_log_set_level(MSP_LEVEL_INFO);
	int ret;
	int id;
	char * envDebug = getenv("MSP_DEBUG");
	char * envFile  = getenv("MSP_DEBUG_FILE"); 
    _debug_init = true;
#ifdef LINUX
    if(execname == NULL)
	{
		static char temp[256];
		prctl(PR_GET_NAME, temp, 0, 0);
		execname = temp;
	}
#else
	execname = getprogname();
#endif
	
	{
		pthread_mutexattr_t   mta;
		pthread_mutexattr_init(&mta);
		/* or PTHREAD_MUTEX_RECURSIVE_NP */
		pthread_mutexattr_settype(&mta, PTHREAD_MUTEX_RECURSIVE);
		pthread_mutex_init(&_module_map_mutex, &mta);
	}
	{
		pthread_mutexattr_t   mta;
		pthread_mutexattr_init(&mta);
		/* or PTHREAD_MUTEX_RECURSIVE_NP */
		pthread_mutexattr_settype(&mta, PTHREAD_MUTEX_RECURSIVE);
		pthread_mutex_init(&_msp_debug_env_map_mutex, &mta);
	}
	{
		pthread_mutexattr_t   mta;
		pthread_mutexattr_init(&mta);
		/* or PTHREAD_MUTEX_RECURSIVE_NP */
		pthread_mutexattr_settype(&mta, PTHREAD_MUTEX_RECURSIVE);
		pthread_mutex_init(&_msp_module_vector_mutex, &mta);
	}

	pid = getpid();
    CheckConfigFile();
	//TODO
	//InitializeCriticalSection (&_msp_module_vector_mutex);

    if(get_config_from_configfile() == -1)
	{    
		MINILOG_DEBUG("read config failed");
	#if 0
        env = getenv("MSP_DEBUG");   
        if (env && strlen(env) != 0)
        {
            strcpy(buf, env);   // There is MSP_DEBUG env. update buf write into file.

            string str = env;
            vector<string> vec = split(str, ',');
        
            for ( it=vec.begin(); it < vec.end(); it++ )
            {
                string node = *it;
                vector<string> v = split(node, ':');
                if (v.size() == 2)
                {
                    pthread_mutex_lock (&_MSP_DEBUG_env_map_mutex);
                    (*_MSP_DEBUG_env_map)[v[0]] = atoi(v[1].c_str());
                    pthread_mutex_unlock (&_MSP_DEBUG_env_map_mutex);

                    if (v[0] == "*")
                        _msp_min_debug_level = atoi(v[1].c_str());
                }
            }
        }else{ 
            strcpy(buf, DEFAULT_MSP_DEBUG_VALUE);
        }

        /*  
         *     #global module Level value
         *     [global]
         *     *:3,MixerAudioPool:5,[moduleName:levelValue]
         *     #module Level value for one PID
         *     [local]
         *     *:3,libVideo:4,[moduleName:levelValue]
         *     *:3,MixerVideoPool:2,[moduleName:levelValue]
         *     ... ...
         * */
        sprintf(conf, "#MSP_LOG's version\n%s\t%d.%s%d\n#global module Level value\n%s\n%s\n#module Level value for one Process\n%s\n",
                    VERSION_CONFIG, MAJOR_VERSION, (MINOR_VERSION / 10) == 0 ? "0" : "", MINOR_VERSION, GLOBAL_CONFIG, buf, LOCAL_CONFIG);

        int     fd;
        do {
            if(( fd = open(MSP_DEBUG_CONF_FILE, O_RDWR | O_CREAT, 0666)) == -1) // open
            {
                fprintf(stderr, "Failed to open/create File: %s\n", MSP_DEBUG_CONF_FILE);
                break;
            }

            if(lockf(fd, F_LOCK, 0) != 0)   // 对正个文件上�?
            {
                fprintf(stderr, "Failed to lock the config file\n");
                break;
            }

            if(write(fd, conf, strlen(conf)) == -1)   // write
                fprintf(stderr, "Failed to write\n");

            if(lockf(fd, F_ULOCK, 0) != 0)  // 解锁.
            {
                fprintf(stderr, "Failed to unlock the config file\n");
                break;
            }

            close(fd);
        } while(0);
		#endif
     }

	if (envDebug && strlen(envDebug) > 0)
    {
    	MINILOG_DEBUG("envDebug=%s", envDebug);
        std::string str = envDebug;
        std::vector<std::string> levels = split(str, ",");
        std::vector<std::string>::iterator it;
    
        for (it = levels.begin(); it < levels.end(); it++ )
        {
            std::vector<std::string> vlevel = split(*it, ":");
            if (vlevel.size() == 2)
            {
				MINILOG_DEBUG(" _debug_env_map name=%s, level=%s\n",vlevel[0].c_str(), vlevel[1].c_str());							
                _debug_env_map[*it] = atoi(vlevel[1].c_str());

                if (vlevel[0] == "*")
                    _debug_global_level = atoi(vlevel[1].c_str());
            }
        	else if (vlevel.size() == 1)
            {
            	int level = atoi((*it).c_str());
            	if (level >= MSP_LEVEL_FATAL || level <= MSP_LEVEL_LOG)
                {
                	_debug_global_level = level;
                }
            	else
                {
                	printf("error log level str:%s\n", (*it).c_str());
                }
            }
        	else
            {
            	printf("error log level str:%s\n", (*it).c_str());
            }
        }
    }
	
	if (envFile && strlen(envFile) > 0)
	{
        _debug_output_file = strdup(envFile);

		MINILOG_DEBUG("output file is :%s", _debug_output_file);
		
        if (NULL == _debug_file_handle)
        {
			if(strcmp(_debug_output_file, "stderr") == 0)
				_debug_file_handle = stderr; 
			else if(strcmp(_debug_output_file, "stdout") == 0)
				_debug_file_handle = stdout; 			
			else 
				_debug_file_handle = fopen(_debug_output_file, "a");
			if(_debug_file_handle == NULL)
			{
				fprintf(stderr, "can not open MSP_DEBUG_OUTPUT file %s\n", _debug_output_file);
			}else{
				MINILOG_INFO("open output file:%s success.", _debug_output_file);
			}
        }
	}
    else
	{
		const char * execname = getprogname();
//        fprintf(stderr, "%s use default output file:%s\n", execname, DEFAULT_OUTPUT_FILE);
        _debug_output_file = strdup(DEFAULT_OUTPUT_FILE);
        if (NULL == _debug_file_handle)
        {
#if 1
			struct stat st;
			if(stat(_debug_output_file, &st) < 0)
			{
#ifndef MACOSX
				fprintf(stderr, "log driver not install, try install log driver");
#endif
			}
#endif
			_debug_file_handle = fopen(_debug_output_file, "a");
        }

    }
}

void msp_log_destroy()
{
	_module_map.clear();
}

void msp_printf(char * sz);
void msp_log_dump_config()
{
	MSPModule * module = NULL;
	LOCK(_module_map_mutex);
	for(std::map<std::string, MSPModule*>::iterator it = _module_map.begin(); it != _module_map.end(); it++)
	{
		char buf[4096];
		snprintf(buf, sizeof(buf), "[MSPLOG]: module=%s, level=%d\n", it->first.c_str(), it->second->level);
		msp_printf(buf);
	}
	UNLOCK(_module_map_mutex);		
}

MSPModule * getModuleAccordingName(const char * name)
{
	if(!name)
	{
		MINILOG_ERROR("invalid module name");
		return NULL;
	}
	if(name[0] == 0)
	{
		MINILOG_ERROR("empty module name");
		return NULL;
	}
    MSPModule * module = NULL;
	LOCK(_module_map_mutex);
	if (!_debug_init)
    {
    	msp_log_init();
    }

#if 0
	std::map<std::string, int>::iterator envIt3; 							
	envIt3 = _debug_env_map.find("test1");
	MINILOG_DEBUG("envIt3->second = %d \n",envIt3->second);
#endif
	
     std::map<std::string, MSPModule*>::iterator modIt = _module_map.find(name);
	if (modIt == _module_map.end())
    {
		//MINILOG_DEBUG("find %s failed\n", name);
    	module = new MSPModule;
    	if (NULL == module)
        {
        	return NULL;
        }
    	module->name = strdup(name);
        std::map<std::string, int>::iterator envIt = _debug_env_map.find(name);
    	if (envIt != _debug_env_map.end())
    	{
            module->level = envIt->second;
    		//MINILOG_DEBUG("_debug_env_map  find! name=%s level=%d\n", module->name, module->level);
    	}
        else
        {
    	    module->level = _debug_global_level;
        }
		_module_map[name] = module;
		MINILOG_DEBUG("create msp module for %s", name);
    }
    else
    {
		MINILOG_DEBUG("find module %s\n", name);
        module = modIt->second;
    }
	UNLOCK(_module_map_mutex);		
    return module;
}

CMSPDebug::CMSPDebug(const char * _name)
{
	assert(_name);
	assert(_name[0]);
	assert(strlen(_name) > 0);
	strncpy(name, _name, sizeof(name));
	MINILOG_DEBUG("CMSPDebug::name=%s", _name);
	m_mod = NULL;
}

CMSPDebug::~CMSPDebug()
{
}

int CMSPDebug::getLevel()
{
	if(m_mod == NULL)
	{
		m_mod = getModuleAccordingName(name);
		assert(m_mod);

		int ret = 0;
		
		#ifdef LINUX
		pthread_t    thread_id;
		#else
		thread_id_t    thread_id;
		#endif
		
		int id;

		LOCK(_msp_debug_env_map_mutex);
		static bool flagThreadReceiveNotificationCreated = false;
		if(!flagThreadReceiveNotificationCreated)
		{
			flagThreadReceiveNotificationCreated = true;
			ret = thread_spawn(&thread_id, receive_notification, (void*)m_mod);
			/*
			* Wait for thread to have registered all ports before starting
			* the clients and the clock.
			*/
			//wait_for_thread
			/* Wait for thread to complete */
			//thread_join(thread_id);
			#ifdef LINUX 
			if(ret == 0)
			{
				pthread_detach(thread_id); // set the new thread detach status.				
			}
			else
			{
				MINILOG_ERROR("creat thread failed.");
				return -1;
			}
			#else
			if(ret == 0)
			{
				pthread_detach(thread_id.tid); // set the new thread detach status.				
			}
			else
			{
				MINILOG_ERROR("creat thread failed.");
				return -1;
			}
			#endif			
		}
		UNLOCK(_msp_debug_env_map_mutex);
		MINILOG_DEBUG("[%s]: find module=%p, this=%p", name, m_mod, this);
	}
	return m_mod->level > _msp_min_debug_level ? m_mod->level : _msp_min_debug_level;
}

void CMSPDebug::setName(const char * _name)
{ 
	strncpy(name, _name, sizeof(name)); 
	MSPModule * m = getModuleAccordingName(name);
	if(m)
	{
		m_mod = m;
		if(strncmp(m->name, _name, sizeof(name)))
			m->name = strdup(name);
	}
};

void CMSPDebug::setLevel(int level)
{
	if(m_mod)
		m_mod->level = level;
}

void msp_get_time(char * time)
{
    unsigned int year, month, day, hour, min, sec, usec;
    struct timeval      tv;
    struct tm       *   tm;

    time[0] = '\0';
    
    gettimeofday (&tv, NULL);
    tm = localtime (&tv.tv_sec);
    sprintf(time, "%4d-%02d-%02d-%02d:%02d:%02d.%06d", tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday, tm->tm_hour, tm->tm_min, tm->tm_sec, tv.tv_usec);
    return;
}


void msp_printf(char * sz)
{
    static int      fd      = -1;
    
    if (_debug_file_handle)
    {
        fwrite(sz, strlen(sz) + 1, 1, _debug_file_handle);
        fflush(_debug_file_handle);
    }
    else
    {
        //fprintf(stderr, sz);
		fwrite(sz, strlen(sz), 1, stderr);
		fflush(stderr);
    }
    return;
}

void msp_debug_log_ex(CMSPDebug * module, unsigned int level, const char * file, const char * function, int line, const char * format, ...)
{
    int     len;
    //char    sz[MSP_LOG_STRING_MAX_LEN_EX] = {0};
    char *  temp = NULL;
    va_list var_args;
    char    sztime[MSP_TIME_LENGTH];

    va_start (var_args, format);
    len = vasprintf (&temp, format, var_args);
    va_end (var_args);

    if (len < 0)
    {
        return;
    }

    msp_get_time(sztime);
	
	char * szbuf = NULL;
	asprintf(&szbuf, "%s%s [%s:%d][%s][%s:%d][%s]: %s%s\n",
		_msp_color_level[level], 
		sztime,
		execname,
		pid,
		module->getName(), 
		basename((char*)file),
//		file,
		line, 
		_msp_level[level], 
		temp,
		MSP_COLOR_NONE);

    msp_printf(szbuf);

    free(temp);
	free(szbuf);
    return;
}

void msp_debug_log(CMSPDebug * module, unsigned int level, const char * file, const char * function, int line, const char * format, ...)
{
    int     len;
    //char    sz[MSP_LOG_STRING_MAX_LEN] = {0};
    char *  temp = NULL;
    va_list var_args;
    char    sztime[MSP_TIME_LENGTH];
    va_start (var_args, format);
    len = vasprintf (&temp, format, var_args);
    va_end (var_args);
	
    if (len < 0)
    {
        return;
    }

    msp_get_time(sztime);

	char * szbuf = NULL;
	asprintf(&szbuf,
			 "%s%s [%d][%s][%s:%d][%s]: %s%s\n",
//			 "%s%s [%s:%d][%s][%s:%d][%s]: %s%s\n",
            _msp_color_level[level],
            sztime,
//			execname,
			pid,
            module->getName(), 
            basename((char*)file),
//            file,
            line, 
			_msp_level[level], 
            temp,
            MSP_COLOR_NONE);

    msp_printf(szbuf);
    free (temp);
	free(szbuf);
    return;
}

void msp_debug_log_o(CMSPDebug * module, unsigned int level, const char * file, const char * function, int line, const char * name, const char * format, ...)
{
    int     len;
    //char    sz[MSP_LOG_STRING_MAX_LEN] = {0};
    char *  temp = NULL;
    va_list var_args;
    char sztime[MSP_TIME_LENGTH];

    va_start (var_args, format);
    len = vasprintf (&temp, format, var_args);
    va_end (var_args);

    if (len < 0)
    {
        return;
    }

    msp_get_time(sztime);

	char * szbuf = NULL;
	asprintf(&szbuf, "%s%s [%s:%d][%s][%s:%d][%s][%s]: %s%s\n",
		_msp_color_level[level], 
		sztime,
		execname,
		pid,
		module->getName(), 
		basename((char*)file),
//		file,
		line, 
		_msp_level[level], 
		name,
		temp,
		MSP_COLOR_NONE);

    msp_printf(szbuf);
    free (temp);
	free(szbuf);
    return;
}

