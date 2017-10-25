#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>

#include "mini_log.h"

#ifdef LINUX
#include <stdarg.h>
#include <libgen.h>
#elif defined(OSX)
#include <libgen.h>
#endif

#define MSP_LOG_STRING_MAX_LEN          4096
#define MSP_LOG_STRING_MAX_LEN_EX       8192
#define MSP_TIME_FORMAT                 "%4d-%02d-%02d-%02d:%02d:%02d.%06d"
#define MSP_TIME_LENGTH                 32
#define MSP_LOG_STRING_LEN              900

#define MSP_COLOR_NONE                  "\033[0m"

#define MAJOR_VERSION   1
#define MINOR_VERSION   0


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

static void split(std::string& s, std::string& delim, std::vector<std::string >* ret)
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

static std::vector<std::string> split(std::string &s, std::string delim) 
{ 
	std::vector<std::string> v; 
	split(s, delim, &v); 
	return v;
}

static void msp_get_time(char * time)
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

static void msp_printf(char * sz)
{
	static int      fd      = -1;

//	printf(sz);
	//fprintf(stderr, sz);
	fwrite(sz, strlen(sz), 1, stdout);
	fflush(stdout);

//	fwrite(sz, strlen(sz), 1, stderr);
//	fflush(stderr);
	return;
}

static unsigned int currentLevel = 0; //MSP_LEVEL_INFO;
MSP_EXPORT void mini_log_set_level(unsigned int level)
{
	if(level >= MSP_LEVEL_FATAL && level <= MSP_LEVEL_LOG)
		currentLevel = level;
	else
	{
		MINILOG_ERROR("mini_log_set_level(): invalid level %d", level); 
	}
}

MSP_EXPORT unsigned int mini_log_get_level()
{
	if(currentLevel <= 0)
	{
		char * env = getenv("MINILOG_LEVEL");
		if(env)
		{
			currentLevel = atoi(env);
		}
		else 
			currentLevel = MSP_LEVEL_INFO;
	}
	return currentLevel;
}

MSP_EXPORT const char * mini_log_sprintf(unsigned int level, const char * file, const char * function, int line, const char * format, ...)
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
		return NULL;
	}

	msp_get_time(sztime);

	char * szbuf = NULL;
	asprintf(&szbuf, "%s%s [%5s][%8s][%12s:%4d]: %s%s\n",
		_msp_color_level[level], 
		sztime,
		_msp_level[level], 
		"", 
		basename((char*)file),
//		file,
		line, 
		temp,
		MSP_COLOR_NONE);
	free (temp);
	return szbuf;
}

MSP_EXPORT void mini_debug_log(unsigned int level, const char * file, const char * function, int line, const char * format, ...)
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
	asprintf(&szbuf, "%s%s [%5s][%8s][%12s:%4d]: %s%s\n",
		_msp_color_level[level], 
		sztime,
		_msp_level[level], 
		"", 
		basename((char*)file),
//		file,
		line, 
		temp,
		MSP_COLOR_NONE);

	msp_printf(szbuf);
	free (temp);
	free(szbuf);
	return;
}

