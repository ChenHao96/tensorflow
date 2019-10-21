import numpy as np

np.random.seed(42)
input_x = np.random.rand(200, 2)
example_y = []

for i in range(len(input_x)):
    result = 3 * input_x[i][0] + 4 * input_x[i][1] - 3.5
    example_y.append(result)

w = np.random.random(2).reshape([2, 1])
b = np.random.random(1)

def perceptron(X, W, b):
    return np.dot(X, W) + b

def error_formula(y, y_hat):
    y = np.float_(y)
    y_hat = np.float_(y_hat)
    return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

learning = 0.01
for _ in range(100):
    for i in range(len(input_x)):
        y_ = perceptron(input_x[i], w, b)
        if y_ > example_y[i]:
            w[0] -= input_x[i][0] * learning
            w[1] -= input_x[i][1] * learning
            b -= learning
        elif y_ < example_y[i]:
            w[0] += input_x[i][0] * learning
            w[1] += input_x[i][1] * learning
            b += learning

print(w)
print(b)
