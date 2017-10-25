#pragma once
#include "msp_log.h"

MSP_EXPORT void mini_log_set_level(unsigned int level);
MSP_EXPORT unsigned int mini_log_get_level();
MSP_EXPORT void mini_debug_log(unsigned int level, const char * file, const char * function, int line, const char * format, ...);
MSP_EXPORT const char * mini_log_sprintf(unsigned int level, const char * file, const char * function, int line, const char * format, ...);
#define MINI_LOG_LEVEL(lvl, ...)      \
{ \
	if(lvl <= mini_log_get_level()) mini_debug_log(lvl, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__);    \
}

#define MINILOG_FATAL(...)          MINI_LOG_LEVEL(MSP_LEVEL_FATAL, __VA_ARGS__)
#define MINILOG_ERROR(...)          MINI_LOG_LEVEL(MSP_LEVEL_ERROR, __VA_ARGS__)
#define MINILOG_WARNING(...)        MINI_LOG_LEVEL(MSP_LEVEL_WARNING, __VA_ARGS__)
#define MINILOG_NOTICE(...)			MINI_LOG_LEVEL(MSP_LEVEL_NOTICE, __VA_ARGS__)
#define MINILOG_INFO(...)           MINI_LOG_LEVEL(MSP_LEVEL_INFO, __VA_ARGS__)
#define MINILOG_DEBUG(...)          MINI_LOG_LEVEL(MSP_LEVEL_DEBUG, __VA_ARGS__)
#define MINILOG_LOG(...)            MINI_LOG_LEVEL(MSP_LEVEL_LOG, __VA_ARGS__)

#ifdef NO_MSP_LOG
#undef MSP_FATAL
#undef MSP_ERROR
#undef MSP_WARNING
#undef MSP_NOTICE
#undef MSP_INFO
#undef MSP_DEBUG
#undef MSP_LOG
#undef MSP_LOG_LEVEL
#undef MSP_GET_LOG_LEVEL

#ifndef NO_MINI_LOG
#define MSP_FATAL(...)          MINI_LOG_LEVEL(MSP_LEVEL_FATAL, __VA_ARGS__)
#define MSP_ERROR(...)          MINI_LOG_LEVEL(MSP_LEVEL_ERROR, __VA_ARGS__)
#define MSP_WARNING(...)        MINI_LOG_LEVEL(MSP_LEVEL_WARNING, __VA_ARGS__)
#define MSP_NOTICE(...)			MINI_LOG_LEVEL(MSP_LEVEL_NOTICE, __VA_ARGS__)
#define MSP_INFO(...)           MINI_LOG_LEVEL(MSP_LEVEL_INFO, __VA_ARGS__)
#define MSP_DEBUG(...)          MINI_LOG_LEVEL(MSP_LEVEL_DEBUG, __VA_ARGS__)
#define MSP_LOG(...)            MINI_LOG_LEVEL(MSP_LEVEL_LOG, __VA_ARGS__)
#define MSP_LOG_LEVEL(mspdebug, ...)			MINI_LOG_LEVEL(__VA_ARGS__)
#else
#define MSP_FATAL(...)         
#define MSP_ERROR(...)          
#define MSP_WARNING(...)        
#define MSP_NOTICE(...)			
#define MSP_INFO(...)           
#define MSP_DEBUG(...)         
#define MSP_LOG(...)            
#define MSP_LOG_LEVEL(mspdebug, ...)			
#endif
#define MSP_GET_LOG_LEVEL		mini_log_get_level()
#endif

