#!/usr/bin/python
# -*- coding: UTF-8 -*-
# tensorflow-1.12-cpu
# Tensorflow 实战Google深度学习框架(第2版) P62-63-64

import tensorflow as tf
from numpy.random import RandomState

# 创建隐藏层变量
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 声明参数占位符
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

# 矩阵乘法(前向传播算法)
# 第一层
a = tf.matmul(x, w1)
# 第二层 获得预测结果
y_hat = tf.matmul(a, w2)

# 将预测数值转换0~1之间
y_hat = tf.sigmoid(y_hat)
# 交叉商算法
cross_entropy = -tf.reduce_mean(
        y * tf.log(tf.clip_by_value(y_hat, 1e-10, 1.0))
        + (1 - y) * tf.log(tf.clip_by_value(1 - y_hat, 1e-10, 1.0)))
# 创建训练推理算法 以1/1000的学习率学习
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = RandomState(1)
dataSetSize = 128

# 生成训练的数据(模拟)
X = rdm.rand(dataSetSize, 2)
# 生成训练数据的结果(模拟)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print(session.run(w1))
    '''
        [[-0.8113182   1.4845988   0.06532937]
        [-2.4427042   0.0992484   0.5912243 ]]
    '''
    print(session.run(w2))
    '''
        [[-0.8113182 ]
         [ 1.4845988 ]
         [ 0.06532937]]
    '''
    STEPS = 5000
    batch_size = 8
    for i in range(STEPS):
        start = (i * batch_size) % dataSetSize
        end = min(start + batch_size, dataSetSize)
        # 推理隐藏层数据
        session.run(train_step, feed_dict={x: X[start:end], y: Y[start:end]})
        if i % 1000 == 0:
            # 查看交叉熵的值 随着推理逐渐变小(交叉熵) 使得隐藏层数据更为精准
            total_cross_entropy = session.run(cross_entropy, feed_dict={x: X, y: Y})
            print("After %d training step(s),cross entropy on all data is %g" % (i, total_cross_entropy))
            '''
                After 0 training step(s),cross entropy on all data is 1.89805
                After 1000 training step(s),cross entropy on all data is 0.655075
                After 2000 training step(s),cross entropy on all data is 0.626172
                After 3000 training step(s),cross entropy on all data is 0.615096
                After 4000 training step(s),cross entropy on all data is 0.610309
            '''
    print(session.run(w1))
    '''
        [[ 0.0247699   0.5694868   1.6921942 ]
         [-2.197735   -0.23668915  1.1143898 ]]
    '''
    print(session.run(w2))
    '''
        [[-0.45544708]
        [ 0.49110925]
         [-0.98110336]]
    '''
