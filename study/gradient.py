#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


y = np.array([14, 20])
x = np.array([[0, 1, 2, 3]])
w = np.array([[0., 1.], [2., 3.], [4., 5.], [6., 7.]])

y_ = np.dot(x, w)
y_ = sigmoid(y_)
loss = np.mean(np.square(y_ - y))
print(loss)

# TODO:?? 求导公式有问题？
for i in range(len(x[0])):
    w_ = loss * y_[0][0] * (1 - y_[0][0]) * x[0][i]
    print(w_)
    w[i][0] += w_
    w_ = loss * y_[0][1] * (1 - y_[0][1]) * x[0][i]
    print(w_)
    w[i][1] += w_

y_ = np.dot(x, w)
y_ = sigmoid(y_)
loss = np.mean(np.square((y_ - y)))
print(loss)
