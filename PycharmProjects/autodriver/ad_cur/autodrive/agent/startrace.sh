#!/usr/bin/env bash

window=
for retry in 1 2 3 4 6 7 8 9 10;
do
    window=`xdotool search --name $1 | head -n 1`
    [ "${window}" != "" ] && break
    sleep 1.
done
echo "window=${window}, name=$1, display=${DISPLAY}"
sleep 0.5
if [ "${window}" == "" ];
then
    echo "no window found for title $1"
    exit 1
fi
if [ "$2" == "quickrace" ];
then
    sleep 0.1
    xdotool key --window $window Return
    sleep 0.1
    xdotool key --window $window Return
    sleep 0.1
    xdotool key --window $window Return
elif [ "$2" == "practice" ];
then
    sleep 0.1
    xdotool key --window $window Return
    for retry in 1 2 3 4 5;
    do
        sleep 0.1
        xdotool key --window $window Down
    done
    sleep 0.1
    xdotool key --window $window Return
    sleep 0.1
    xdotool key --window $window Return
fi