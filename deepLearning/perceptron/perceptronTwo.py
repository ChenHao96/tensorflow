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

weights = np.random.rand(1, 2)[0]
bias = np.random.rand(1, 1)[0]
print(weights, bias)


# Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Output (prediction) formula
def output_formula(features, weights, bias):
    if len(features.shape) == 1:
        return sigmoid(np.dot(weights, features) + bias)
    else:
        result = []
        for i in range(len(features)):
            result.append(sigmoid(np.dot(weights, features[i]) + bias))
        return np.array(result)


# Error (log-loss) formula
def error_formula(y, y_hat):
    y = np.float_(y)
    y_hat = np.float_(y_hat)
    return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


# Gradient descent step
def update_weights(x, y, weights, bias, learn_rate):
    y_hat = output_formula(x, weights, bias)
    ar = learn_rate * (y - y_hat)
    bias += ar
    for i in range(len(x)):
        weights[i] += ar * x[i]
    return weights, bias


errors = []
epochs = 100
learn_rate = 0.1
for e in range(epochs):
    for x, y in zip(input_x, example_y):
        weights, bias = update_weights(x, y, weights, bias, learn_rate)

    y_hat = output_formula(input_x, weights, bias)
    errors.append(error_formula(example_y, y_hat))

print(weights, bias)

model = []
model.extend(weights)
model.append(bias)
np.savetxt("./model.txt", model)
