#include "stdafx.h"
#include "utils.h"

bool getEnvBool(const char * key, bool defaultValue /* = false */)
{
	const char * v = getenv(key);
	if(!v)
		return defaultValue;
	QString str = v;

	return str.toLower() == "yes" || str == "1";
}


int32_t getEnvInt(const char * key, int defaultValue /* = 0 */)
{
	const char * v = getenv(key);
	if(!v)
		return defaultValue;
	QString str = v;
	return str.toInt();
}

uint32_t getEnvUInt(const char * key, uint32_t defaultValue /* = 0 */)
{
	const char * v = getenv(key);
	if(!v)
		return defaultValue;
	QString str = v;
	return str.toUInt();
}

const std::string getEnvString(const char* key, const char * defaultValue /* = NULL */)
{
	const char * v = getenv(key);
	if(!v)
		return defaultValue;
	return v;
}