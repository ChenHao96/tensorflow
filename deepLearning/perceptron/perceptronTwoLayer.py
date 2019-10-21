#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

# todo: add some one-hot-data
input_x = np.array([[0, 0]])
# todo: add flag data
example_y = []


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)


def error_formula(y, y_hat):
    y = np.float_(y)
    y_hat = np.float_(y_hat)
    return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def error_term_formula(x, y, y_hat):
    return (y - y_hat) * sigmoid_prime(x)
    # return (y - y_hat) * y_hat * (1 - y_hat)


n_records, n_features = input_x.shape
weights = np.random.normal(scale=1 / n_features ** .5, size=n_features)

epochs = 1000
learn_rate = 0.5
for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(input_x, example_y):
        output = sigmoid(np.dot(x, weights))
        error_term = error_term_formula(x, y, output)
        del_w += error_term * x

    weights += learn_rate * del_w / n_records

np.savetxt("./model.txt", weights)
