#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

np.random.seed(42)
input_x = np.random.rand(200, 2)
example_y = []

for i in range(len(input_x)):
    result = 3 * input_x[i][0] + 4 * input_x[i][1] - 3.5
    if result >= 0:
        example_y.append(1)
    else:
        example_y.append(0)

np.savetxt("./train-datas.txt", input_x)
np.savetxt("./train-labels.txt", example_y)

W = np.random.rand(1, 2)[0]
b = np.random.rand(1, 1)[0]
print(W, b)


def step_function(t):
    if t >= 0:
        return 1
    return 0


def perceptron(X, W, b):
    return np.dot(W, X) + b


learn_rate = 0.1
for t in range(200):
    for i in range(len(input_x)):
        y_hat = step_function(perceptron(input_x[i], W, b))

        if example_y[i] - y_hat > 0:
            result = 1
        elif example_y[i] - y_hat < 0:
            result = -1
        else:
            continue

        W[0] += input_x[i][0] * learn_rate
        W[1] += input_x[i][1] * learn_rate
        b += learn_rate * result

print(W, b)

model = []
model.extend(W)
model.append(b)
np.savetxt("./model.txt", model)
