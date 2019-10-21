#!/usr/bin/python
# -*- coding: UTF-8 -*-

import threading
import tkinter as tk

from raspberrypi.video import *


def thread_method():
    image_file_index = 0
    while True:
        canvas.create_image(wx / 2, wh / 2, image=image_files[image_file_index])
        image_file_index ^= 1
        record_video_capture(capture, image_files, image_file_index)


image_files = [None, None]
window = tk.Tk()
window.title("本地设备视频")

capture = cv2.VideoCapture(0)
wx, wh = record_video_capture(capture, image_files, 0)

canvas = tk.Canvas(window, height=wh, width=wx)
canvas.pack()

thread = threading.Thread(target=thread_method)
thread.setDaemon(True)
thread.start()

window.mainloop()
