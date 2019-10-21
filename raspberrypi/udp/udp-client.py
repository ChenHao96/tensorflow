#!/usr/bin/python
# -*- coding: UTF-8 -*-

import socket

from raspberrypi.ip import get_ip_address

UDP_STATIC_PORT = 34381

message = "hello world".encode()

address = get_ip_address()
address = address[:address.rindex(".")]+".255"

udp_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    udp_client.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    udp_client.sendto(message, (address, UDP_STATIC_PORT))
finally:
    udp_client.close()
