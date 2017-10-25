#!/bin/bash

set -eu
#ident=$1
#port=$2
#window_title=$3
#
#export LOCAL_CONF_DIR=${HOME}/.torcs_${ident}
#
#${HOME}/torcs/bin/torcs -nofuel -nodamage -nolaptime -title ${window_title} -p ${port}

for retry in 1 2 3 4 6 7 8 9 10;
do
    window=`xdotool search --name $1 | head -n 1`
    [ '${window}' != '' ] && break
    sleep 0.1
done


if [ $# -ge 3 ];
 then
    for retry in 1 2 3 4 5 6 7 8 9 10:
    do
        xdotool windowmove $window $2 $3
        [ $? -eq 0 ] && break
        sleep 0.1
    done
 fi
sleep 0.1
xdotool key --window $window Return
sleep 0.1
xdotool key --window $window Return
sleep 0.1
xdotool key --window $window Up
sleep 0.1
xdotool key --window $window Up
sleep 0.1
xdotool key --window $window Return
sleep 0.1
xdotool key --window $window Return
# Uncomment for using vision as input
sleep 0.1
xdotool key --window $window F2
sleep 0.1
xdotool key --window $window 0
