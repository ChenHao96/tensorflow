#!/usr/bin/python
# -*- coding: UTF-8 -*-
# tensorflow-1.12-cpu
# Tensorflow 实战Google深度学习框架(第2版) P37-38

import tensorflow as tf

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b
session = tf.Session()
print(session.run(result))
'''
[3., 5.]
'''
session.close()
