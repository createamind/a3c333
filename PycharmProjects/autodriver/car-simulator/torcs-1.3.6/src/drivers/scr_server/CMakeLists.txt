cmake_minimum_required(VERSION 3.8)
project(scr_server)

set(CMAKE_CXX_STANDARD 11)
add_definitions(-DLINUX)

set(TOPDIR ${CMAKE_SOURCE_DIR}/../../../)
include_directories(utils
        ${TOPDIR}/src/interfaces
        ${TOPDIR}/src/libs/tgf
        ${TOPDIR}/src/libs/math
        ${TOPDIR}/src/linux
        ${TOPDIR}/export/include
        $ENV{HOME}/anaconda3/include/qt
        )
message(${TOPDIR})
set(SOURCE_FILES
        utils/ice_server.cpp
        utils/ice_server.h
        utils/liblog.cpp
        utils/mini_log.cpp
        utils/mini_log.h
        utils/msp_log.h
        utils/race.cpp
        utils/race.h
        utils/race.ice
        CarControl.cpp
        CarControl.h
        utils/ObstacleSensors.cpp
        utils/ObstacleSensors.h
        scr_server.cpp
        utils/sensors.cpp
        utils/sensors.h
        SimpleParser.cpp
        SimpleParser.h utils/utils.h utils/stdafx.h utils/race_bot.cpp
        utils/utils_environment.cpp
        utils/utils_string.cpp
        utils/race_bot.h)

add_library(scr_server ${SOURCE_FILES})