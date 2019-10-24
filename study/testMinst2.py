# use tensorflow-2.0

import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers

(xs, ys), (x_val, y_val) = datasets.mnist.load_data()
xs = tf.convert_to_tensor(xs, dtype=tf.float32) / 255.
ys = tf.convert_to_tensor(ys, dtype=tf.int32)
ys = tf.one_hot(ys, depth=10)
db = tf.data.Dataset.from_tensor_slices((xs, ys))
db = db.batch(200)

model = tf.keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)])

optimizer = optimizers.SGD(learning_rate=0.001)

for epoch in range(10):
    for step, (x, y) in enumerate(db):
        with tf.GradientTape() as tape:
            x = tf.reshape(x, (-1, 28 * 28))
            out = model(x)
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, "loss:", loss.numpy())
