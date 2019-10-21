#!/usr/bin/python
# -*- coding: UTF-8 -*-

import socket

TCP_STATIC_PORT = 34381
BUFF_SIZE = 1024

tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    tcp_server.bind(("", TCP_STATIC_PORT))
    tcp_server.listen(1)
    client, address = tcp_server.accept()
    client.send("Hello")
    print(client.recv(BUFF_SIZE))
    client.close()
finally:
    tcp_server.close()
