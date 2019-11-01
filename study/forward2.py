# use tensorflow-2.0

import os

import tensorflow as tf
from tensorflow.keras import datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def mnist_dataset(batch):
    (train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
    train_x = tf.convert_to_tensor(train_x, dtype=tf.float32) / 255.
    train_y = tf.convert_to_tensor(train_y, dtype=tf.int32)
    train_y = tf.one_hot(train_y, depth=10)
    trains = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    trains = trains.shuffle(train_y.shape[0]).batch(batch)

    test_x = tf.convert_to_tensor(test_x, dtype=tf.float32) / 255.
    test_y = tf.convert_to_tensor(test_y, dtype=tf.int64)
    tests = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    tests = tests.shuffle(test_y.shape[0]).batch(batch)
    return trains, tests


train_dataset, test_dataset = mnist_dataset(128)

w1 = tf.Variable(tf.random.truncated_normal([784, 512], stddev=0.1))
b1 = tf.Variable(tf.zeros([512]))

w2 = tf.Variable(tf.random.truncated_normal([512, 256], stddev=0.1))
b2 = tf.Variable(tf.ones([256]))

w3 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b3 = tf.Variable(tf.zeros([128]))

w4 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b4 = tf.Variable(tf.ones([10]))

train_vars = [w1, b1, w2, b2, w3, b3, w4, b4]

learning = 0.001

for epoch in range(20):
    for step, (x, y) in enumerate(train_dataset):
        x = tf.reshape(x, [-1, 28 * 28])

        with tf.GradientTape() as tape:
            h1 = tf.nn.relu(x @ w1 + b1)
            h2 = tf.nn.relu(h1 @ w2 + b2)
            h3 = tf.nn.relu(h2 @ w3 + b3)
            out = h3 @ w4 + b4
            loss = tf.reduce_mean(tf.square(out - y))

        grades = tape.gradient(loss, train_vars)

        for index in range(len(grades)):
            train_vars[index].assign_sub(learning * grades[index])

        if step % 100 == 0:
            print(epoch, step, "loss:", loss.numpy())

    total_correct, total_number = 0, 0
    for step, (x, y) in enumerate(test_dataset):
        x = tf.reshape(x, [-1, 28 * 28])
        h1 = tf.nn.relu(x @ w1 + b1)
        h2 = tf.nn.relu(h1 @ w2 + b2)
        h3 = tf.nn.relu(h2 @ w3 + b3)
        out = h3 @ w4 + b4
        pred = tf.argmax(tf.nn.softmax(out, axis=1), axis=1)
        total_correct += int(tf.reduce_sum(tf.cast(tf.equal(pred, y), dtype=tf.int32)))
        total_number += x.shape[0]

    print("test correct:", total_correct / total_number)
