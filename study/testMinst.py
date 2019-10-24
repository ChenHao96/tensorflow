#!/usr/bin/python
# -*- coding: UTF-8 -*-
# tensorflow-1.12-cpu
# http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html

import tensorflow as tf

import study.input_data as input_data

dataSet = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for i in range(1000):
        batch_xs, batch_ys = dataSet.train.next_batch(100)
        session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), "float"))
    print(session.run(accuracy, feed_dict={x: dataSet.test.images, y_: dataSet.test.labels}))
