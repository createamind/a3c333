##############################################################################
#
#    file                 : Makefile
#    created              : Sat Mar 18 23:38:24 CET 2000
#    copyright            : (C) 2000 by Eric Espie                         
#    email                : torcs@free.fr   
#    version              : $Id: Makefile,v 1.10 2003/05/18 20:41:27 torcs Exp $                                  
#
##############################################################################
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
##############################################################################


SOLIBRARY = libiceapi.so

SOLIBDIR  = .

SOURCES = utils/race.cpp
SOURCES += utils/utils_string.cpp utils/utils_environment.cpp \
            utils/liblog.cpp utils/mini_log.cpp \
            utils/ice_server.cpp utils/bot_hook.cpp utils/bot.cpp \
            utils/recycle_queue.cpp utils/race_server.cpp \
            utils/sensors.cpp utils/ObstacleSensors.cpp utils/data.cpp



EXPDIR  = include

EXPORTS = utils/ice_server.h

SHIPDIR = config

SHIP    = params.dtd

src_PKGFILES	= $(shell find * -maxdepth 0 -type f -print)

src_PKGDIR	= ${PACKAGE}-${VERSION}/$(subst ${TORCS_BASE},,$(shell pwd))

COMPILFLAGS += -DLINUX -std=c++11 -Iutils -I/usr/include/x86_64-linux-gnu/qt5 -I. -DICE_CPP11_MAPPING
LIBS += -lIce++11 -lbsd -lQt5Core
LDFLAGS += -L/usr/lib/x86_64-linux-gnu/
include ${MAKE_DEFAULT}



