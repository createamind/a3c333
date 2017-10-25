#include <string>
#include <stdio.h>
#include <stdarg.h>
#include <memory.h>
const std::string stdsprintf(const char * msg, ...)
{
    va_list va;
    va_start(va, msg);
    char *buf = NULL;
    vasprintf(&buf, msg, va);
    std::string ret = buf;
    free(buf);
    va_end(va);
    return ret;
}