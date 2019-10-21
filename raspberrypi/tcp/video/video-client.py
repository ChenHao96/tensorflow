#!/usr/bin/python
# -*- coding: UTF-8 -*-

import socket
import threading
import tkinter

from raspberrypi.ip import get_ip_address
from raspberrypi.tcp.video.image_process import *

TCP_STATIC_PORT = 34381

window = tkinter.Tk()
window.title("远程设备视频")


def read_image_socket(address):
    canvas = None
    init_video_canvas = True
    print("connect %s:%s read video data..." % (address, TCP_STATIC_PORT))
    tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        tcp_client.connect((address, TCP_STATIC_PORT))
        while True:
            head = tcp_client.recv(DATA_HEAD_LENGTH)
            read_length = data_receive_length(head)
            data = tcp_client.recv(read_length)
            image_file = decode_image_array(data, head)
            if image_file is not None:
                wh, wx = image_file.height(), image_file.width()
                if init_video_canvas:
                    canvas = tkinter.Canvas(window, height=wh, width=wx)
                    init_video_canvas = False
                    canvas.pack()
                canvas.create_image(wx / 2, wh / 2, image=image_file)

    finally:
        tcp_client.close()


def thread_method():
    read_image_socket(get_ip_address())


thread = threading.Thread(target=thread_method)
thread.setDaemon(True)
thread.start()

window.mainloop()
