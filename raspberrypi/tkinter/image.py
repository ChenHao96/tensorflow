#!/usr/bin/python
# -*- coding: UTF-8 -*-

from tkinter import *

root = Tk()
root.title("Tensorflow")
root.resizable(width=False, height=False)

canvasWidth = 800
canvasHeight = 600

canvas = Canvas(root, width=canvasWidth, height=canvasHeight, bg='white')
canvas.pack(fill=BOTH, expand=YES)

# 只支持gif格式的图片
photo = PhotoImage(file="tensorflow.gif")

canvas.create_image(canvasWidth / 2, canvasHeight / 2, image=photo, tag="pic")

root.mainloop()
