import numpy as np
import numpy.random as rng
import theano
import theano.tensor as T

feats = 3
train_size = 10
data = (rng.randn(train_size, feats).astype(np.float32), rng.randint(size=train_size, low=0, high=2).astype(np.float32))

x = T.matrix("x")
y = T.vector("y")

learn_rate = 0.01
bias = theano.shared(0., name="b")
weights = theano.shared(rng.randn(feats), name="w")

perceptron = T.dot(x, weights) + bias
sigmoid = 1 / (1 + T.exp(-perceptron))
cross_entropy = (-y * T.log(sigmoid) - (1 - y) * T.log(1 - sigmoid)).mean()
regularization_l2 = cross_entropy + learn_rate * (weights ** 2).sum()

gw, gb = T.grad(regularization_l2, [weights, bias])
train = theano.function(inputs=[x, y], outputs=[sigmoid > 0.5, cross_entropy],
                        updates=((weights, weights - learn_rate * gw), (bias, bias - learn_rate * gb)))

train_steps = 1000
for i in range(train_steps):
    _, err = train(data[0], data[1])
    print(err)
