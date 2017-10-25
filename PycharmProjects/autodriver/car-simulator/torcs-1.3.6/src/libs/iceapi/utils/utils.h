#pragma once

#include <string>
const std::string stdsprintf(const char * msg, ...);
bool getEnvBool(const char * key, bool defaultValue = false);
const std::string getEnvString(const char* key, const char * defaultValue = NULL);
int32_t getEnvInt(const char * key, int defaultValue = 0);
uint32_t getEnvUInt(const char * key, uint32_t defaultValue = 0);
#define ARRAYSIZE(x) (sizeof(x)/sizeof(x[0]))
