#!/usr/bin/python
# -*- coding: UTF-8 -*-

import socket

from raspberrypi.ip import get_ip_address

TCP_STATIC_PORT = 34381
BUFF_SIZE = 1024

tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serverAddress = get_ip_address()

try:
    tcp_client.connect((serverAddress, TCP_STATIC_PORT))
    print(tcp_client.recv(BUFF_SIZE))
    tcp_client.send("Hi")
finally:
    tcp_client.close()
