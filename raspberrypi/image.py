#!/usr/bin/python
# -*- coding: UTF-8 -*-
from PIL import Image, ImageTk


def record_image_array(image_array, files, index):
    img = Image.fromarray(image_array)
    image_file = ImageTk.PhotoImage(img)
    files[index] = image_file
    return image_file.width(), image_file.height()
