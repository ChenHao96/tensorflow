#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import numpy as np

input_x = np.loadtxt("./train-datas.txt")
example_y = np.loadtxt("./train-labels.txt")

model = np.loadtxt("./model.txt")
W, b = model[:len(model) - 1], model[len(model) - 1:]

for i in range(len(input_x)):
    if example_y[i] == 1:
        plt.plot(input_x[i][0], input_x[i][1], ".b")
    else:
        plt.plot(input_x[i][0], input_x[i][1], ".r")

x_ = np.linspace(0, 1, len(input_x))
y_ = -W[0] / W[1] * x_ - b / W[1]
plt.plot(x_, y_, "-g")
plt.show()
