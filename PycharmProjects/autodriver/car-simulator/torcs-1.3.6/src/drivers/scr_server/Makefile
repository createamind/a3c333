##############################################################################
#
#    file                 : Makefile
#    created              : Tue Dec 18 23:40:54 CET 2007
#    copyright            : (C) 2002 Daniele Loiacono
#
##############################################################################
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
##############################################################################

ROBOT       = scr_server
MODULE      = ${ROBOT}.so
MODULEDIR   = drivers/${ROBOT}
SOURCES     = ${ROBOT}.cpp
#sensors.cpp SimpleParser.cpp CarControl.cpp ObstacleSensors.cpp
SOURCES += utils/race_bot_scr.cpp
#SOURCES += utils/utils_string.cpp utils/utils_environment.cpp \
#            utils/liblog.cpp utils/mini_log.cpp \
#			utils/ice_server.cpp utils/race_bot.cpp

SHIPDIR     = drivers/${ROBOT}
SHIP        = ${ROBOT}.xml 
SHIPSUBDIRS = 0 1 2 3 4 5 6 7 8 9

PKGSUBDIRS  = ${SHIPSUBDIRS}
src-robots-scr_server_PKGFILES = $(shell find * -maxdepth 0 -type f -print)
src-robots-scr_server_PKGDIR   = ${PACKAGE}-${VERSION}/$(subst ${TORCS_BASE},,$(shell pwd))
COMPILFLAGS += -DLINUX -std=c++11 -Iutils -I/usr/include/x86_64-linux-gnu/qt5 -I. -DICE_CPP11_MAPPING
LIBS += -lIce++11 -lbsd -lQt5Core -liceapi
include ${MAKE_DEFAULT}

