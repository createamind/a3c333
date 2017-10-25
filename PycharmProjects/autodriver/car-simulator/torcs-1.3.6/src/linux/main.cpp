/***************************************************************************

    file                 : main.cpp
    created              : Sat Mar 18 23:54:30 CET 2000
    copyright            : (C) 2000 by Eric Espie
    email                : torcs@free.fr
    version              : $Id: main.cpp,v 1.14.2.3 2012/06/01 01:59:42 berniw Exp $

 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
#include "utils/msp_log.h"
MSP_MODULE_DECLARE("race");

#include <stdlib.h>
#include <GL/glut.h>

#include <tgfclient.h>
#include <client.h>

#include "linuxspec.h"
#include <raceinit.h>

extern bool bKeepModules;

char* window_title = "torcs";
char * pid_file = nullptr;
#include "../libs/iceapi/utils/ice_server.h"

static void
init_args(int argc, char **argv, const char **raceconfig)
{
	int i;
	char *buf;
    
    setNoisy(false);
    setVersion("2013");

	i = 1;

	msp_log_init();
	setUDPListenPort(30000);
	while(i < argc) {
		if(strncmp(argv[i], "-l", 2) == 0) {
			i++;

			if(i < argc) {
				buf = (char *)malloc(strlen(argv[i]) + 2);
				sprintf(buf, "%s/", argv[i]);
				SetLocalDir(buf);
				free(buf);
				i++;
			}
		} else if(strncmp(argv[i], "-L", 2) == 0) {
			i++;

			if(i < argc) {
				buf = (char *)malloc(strlen(argv[i]) + 2);
				sprintf(buf, "%s/", argv[i]);
				SetLibDir(buf);
				free(buf);
				i++;
			}
		} else if(strncmp(argv[i], "-D", 2) == 0) {
			i++;

			if(i < argc) {
				buf = (char *)malloc(strlen(argv[i]) + 2);
				sprintf(buf, "%s/", argv[i]);
				SetDataDir(buf);
				free(buf);
				i++;
			}
		} else if(strncmp(argv[i], "-s", 2) == 0) {
			i++;
			SetSingleTextureMode();
		} else if (strncmp(argv[i], "-timeout", 8) == 0) {
		    i++;
		    if (i < argc) {
			long int t;
			sscanf(argv[i],"%ld",&t);
			setTimeout(t);
			printf("UDP Timeout set to %ld 10E-6 seconds.\n",t);
			i++;
		    }
		} else if (strncmp(argv[i], "-nodamage", 9) == 0) {
		    i++;
		    setDamageLimit(false);
		    printf("Car damages disabled!\n");
		} else if (strncmp(argv[i], "-nofuel", 7) == 0) {
		    i++;
		    setFuelConsumption(false);
		    printf("Fuel consumption disabled!\n");
		} else if (strncmp(argv[i], "-noisy", 6) == 0) {
		    i++;
		    setNoisy(true);
		    printf("Noisy Sensors!\n");
		} else if (strncmp(argv[i], "-ver", 4) == 0) {
		    i++;
		    if (i < argc) {
					setVersion(argv[i]);
		    		printf("Set version: \"%s\"\n",getVersion());
		    		i++;
		    }
		} else if (strncmp(argv[i], "-nolaptime", 10) == 0) {
		    i++;
		    setLaptimeLimit(false);
		    printf("Laptime limit disabled!\n");   
		} else if(strncmp(argv[i], "-k", 2) == 0) {
			i++;
			// Keep modules in memory (for valgrind)
			printf("Unloading modules disabled, just intended for valgrind runs.\n");
			bKeepModules = true;
#ifndef FREEGLUT
		} else if(strncmp(argv[i], "-m", 2) == 0) {
			i++;
			GfuiMouseSetHWPresent(); /* allow the hardware cursor */
#endif
		} else if(strncmp(argv[i], "-r", 2) == 0) {
			i++;
			*raceconfig = "";

			if(i < argc) {
				*raceconfig = argv[i];
				i++;
			}

			if((strlen(*raceconfig) == 0) || (strstr(*raceconfig, ".xml") == 0)) {
				printf("Please specify a race configuration xml when using -r\n");
				exit(1);
			}
		} else if (strncmp(argv[i], "-port", 5) == 0) {
            i++;
            setUDPListenPort(atoi(argv[i]));
            i++;
            printf("UDP Listen Port set to %d!\n", getUDPListenPort());
        } else if (strncmp(argv[i], "-pidfile", 8) == 0) {
            i++;
            pid_file = argv[i];
            i++;
            FILE * fp = fopen(pid_file, "w+");
            if (fp == NULL) {
                printf("Can not open pidfile %s\n", pid_file);
                exit(1);
            }
            else {
                fprintf(fp, "%d\n", getpid());
                fclose(fp);
            }
        } else if (strncmp(argv[i], "-title", 6) == 0) {
            i++;
            printf("Window title set to %s\n", argv[i]);
            window_title = argv[i];
            i++;

        }
		else {
			i++;		/* ignore bad args */
		}
	}

#ifdef FREEGLUT
	GfuiMouseSetHWPresent(); /* allow the hardware cursor (freeglut pb ?) */
#endif
}

/*
 * Function
 *	main
 *
 * Description
 *	LINUX entry point of TORCS
 *
 * Parameters
 *
 *
 * Return
 *
 *
 * Remarks
 *
 */
int
main(int argc, char *argv[])
{
	const char *raceconfig = "";
	MSP_NOTICE("torcs start, pid=%d", getpid())
	init_args(argc, argv, &raceconfig);
	LinuxSpecInit();			/* init specific linux functions */
    if (getUDPListenPort() > 0) {
		iceEnv = std::shared_ptr<IceEnv>(new IceEnv);
		InitParams params;
		params.maxBotCount = 10;
		params.ice_host = getEnvString("ICE_LISTEN_HOST", "127.0.0.1");
		params.ice_port = getUDPListenPort();
		params.ice_timeout = 3000;
		iceEnv->init(params);
        auto funcExit = []() {
            if (iceEnv) {
                iceEnv->close();
                iceEnv = nullptr;
            }
			if (pid_file) {
				unlink(pid_file);
			}
			MSP_NOTICE("torcs exit, pid=%d", getpid())
        };
        atexit(funcExit);
	}
	if(strlen(raceconfig) == 0) {
		GfScrInit(argc, argv);	/* init screen */
		TorcsEntry();			/* launch TORCS */
		glutSetWindowTitle(window_title);
		glutMainLoop();			/* event loop of glut */
	} else {
		// Run race from console, no Window, no OpenGL/OpenAL etc.
		// Thought for blind scripted AI training
		ReRunRaceOnConsole(raceconfig);
		MSP_INFO("race exit main");
	}
	return 0;					/* just for the compiler, never reached */
}

