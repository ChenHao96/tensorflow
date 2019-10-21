#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tkinter.messagebox as mbox
from tkinter import Tk


def on_closing():
    if mbox.askokcancel("Quit", "Do you want to quit?"):
        top.destroy()


top = Tk()
top.protocol("WM_DELETE_WINDOW", on_closing)
top.mainloop()
