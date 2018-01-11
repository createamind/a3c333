mknod /dev/msplog c 255 0
chmod 666 /dev/msplog
insmod vmx_log_driver.ko
