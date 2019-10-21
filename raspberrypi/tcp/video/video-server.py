#!/usr/bin/python
# -*- coding: UTF-8 -*-

import socket
import threading as thread

from raspberrypi.tcp.video.image_process import encode_image_array
from raspberrypi.video import *

BUFFER_SIZE = 65535
TCP_STATIC_PORT = 34381
video_data = None


def open_video():
    global video_data
    capture = cv2.VideoCapture(0)
    while True:
        success, data = record_video_capture_2_array(capture, False)
        if success:
            video_data = data


def thread_socket(listen=3):
    tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        tcp_server.bind(("", TCP_STATIC_PORT))
        tcp_server.listen(listen)
        while True:
            client, address = tcp_server.accept()
            process_thread = thread.Thread(target=process_client, args=client)
            process_thread.setDaemon(True)
            process_thread.start()
    finally:
        tcp_server.close()


def process_client(client):
    if client is not None:
        try:
            while True:
                data = encode_image_array(video_data)
                if data is not None:
                    client.send(data)
        finally:
            client.close()


video_thread = thread.Thread(target=open_video)
video_thread.setDaemon(True)
socket_thread = thread.Thread(target=thread_socket, args=5)
socket_thread.setDaemon(True)

video_thread.start()
socket_thread.start()
