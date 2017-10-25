##############################################################################
#
#    file                 : Makefile
#    created              : Sat Mar 18 23:53:52 CET 2000
#    copyright            : (C) 2000 by Eric Espie                         
#    email                : torcs@free.fr   
#    version              : $Id: Makefile,v 1.12.2.1 2011/12/28 15:05:11 berniw Exp $                                  
#
##############################################################################
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
##############################################################################

TOOLSDIR    = 

TOOLSCRIPT  = torcs

PROGRAM = torcs-bin

SOURCES = main.cpp linuxspec.cpp
#SOURCES += ../drivers/scr_server/utils/ice_server.cpp \
#           ../drivers/scr_server/utils/race.cpp \
#           ../drivers/scr_server/utils/msp_log.cpp \
#           ../drivers/scr_server/utils/utils_string.cpp

LDFLAGS = -rdynamic

SOLIBS  =  -lclient       \
	   -lconfscreens  \
	   -ltgf          \
	   -ltgfclient    \
	   -ltxml	  \
	   -lplibul	  \
	   -lraceengine   \
	   -lmusicplayer  \
	   -lracescreens  \
	   -lrobottools   \
	   -llearning \
	   -lpthread \
       -liceapi \
	   -lIce++11
COMPILFLAGS += -I/usr/include/x86_64-linux-gnu/qt5 -std=c++11 -fPIC -DICE_CPP11_MAPPING

EXPDIR  = include

EXPORTS = osspec.h

src_PKGFILES	= $(shell find * -maxdepth 0 -type f -print)

src_PKGDIR	= ${PACKAGE}-${VERSION}/$(subst ${TORCS_BASE},,$(shell pwd))
COMPILFLAGS += -DLINUX -I../libs/iceapi
include ${MAKE_DEFAULT}




