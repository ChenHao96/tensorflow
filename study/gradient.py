#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


y = np.array([14, 20])
x = np.array([[0, 1, 2, 3]])
w = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])

y_ = np.dot(x, w)
print(y_)

y_ = sigmoid(y_)

loss = np.mean(np.square((y_ - y)))
print(loss)

# TODO:gradient

