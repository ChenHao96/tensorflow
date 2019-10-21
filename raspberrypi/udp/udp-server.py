#!/usr/bin/python
# -*- coding: UTF-8 -*-

import socket

UDP_STATIC_PORT = 34381

udp_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    udp_server.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    udp_server.bind(("", UDP_STATIC_PORT))

    message, address = udp_server.recvfrom(1024)
    print(message.decode(), address)
finally:
    udp_server.close()
