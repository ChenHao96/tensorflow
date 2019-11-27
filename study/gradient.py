#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    result = sigmoid(z)
    return result * (1 - result)


y = np.array([14, 20])
x = np.array([[0, 1, 2, 3]])
w = np.array([[0., 1.], [2., 3.], [4., 5.], [6., 7.]])

z = np.dot(x, w)
z = sigmoid(z)
loss = np.mean(np.square(z - y))
print(loss)
