# use tensorflow-2.0
import os

import tensorflow as tf
from tensorflow.keras import datasets, Sequential, layers, metrics, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.one_hot(y, depth=100)
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batchSize = 128
(x, y), (val_x, val_y) = datasets.cifar100.load_data()
print(x.shape, x.min(), x.max())
print(y.shape, y.min(), y.max())

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(10000).batch(batchSize)

val_db = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_db = val_db.map(preprocess).batch(batchSize)

cnn_net = Sequential([
    layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
])
cnn_net.build(input_shape=[None, 32, 32, 3])
cnn_net.summary()

fc_input_shape = 512
fc_net = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(100)
])
fc_net.build(input_shape=[None, fc_input_shape])
fc_net.summary()

lossMean = metrics.Mean()
accuracy = metrics.Accuracy()

optimizer = optimizers.Adam(lr=1e-4)
variables = cnn_net.trainable_variables + fc_net.trainable_variables

# cnn_net.load_weights("checkpoint/testCNN.cnn")
# fc_net.load_weights("checkpoint/testCNN.fc")

for epoch in range(100):
    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out = cnn_net(x)
            out = tf.reshape(out, (-1, fc_input_shape))
            logits = fc_net(out)
            loss = tf.losses.categorical_crossentropy(y, logits, from_logits=True)
            loss = tf.reduce_mean(loss)
            lossMean.update_state(loss)
        gradient = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradient, variables))

        if step % 100 == 0:
            print(epoch, step, "train loss:", lossMean.result().numpy())
            lossMean.reset_states()

    if epoch % 10 == 0:
        accuracy.reset_states()
        for (x, y) in val_db:
            out = cnn_net(x)
            out = tf.reshape(out, (-1, fc_input_shape))
            logits = fc_net(out)
            pro = tf.nn.softmax(logits, axis=1)
            pro = tf.cast(tf.argmax(pro, axis=1), dtype=tf.int32)
            accuracy.update_state(y, pro)
        print(epoch, "test correct:", accuracy.result().numpy())
        cnn_net.save_weights("checkpoint/testCNN.cnn")
        fc_net.save_weights("checkpoint/testCNN.fc")
