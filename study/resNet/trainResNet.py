# use tensorflow-2.0
import os

import tensorflow as tf
from resNet import ResNet
from tensorflow.keras import datasets, optimizers, metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)


def preprocess(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


classes = 100
(x, y), (val_x, val_y) = datasets.cifar100.load_data()

input_shape = (None, 32, 32, 3)
y = tf.squeeze(y, axis=1)
val_y = tf.squeeze(val_y, axis=1)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(10000).batch(250)

val_db = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_db = val_db.map(preprocess).batch(250)

model = ResNet(layer_dims=[2, 2, 2, 2], classes_num=classes)
model.build(input_shape=input_shape)
model.summary()

optimizer = optimizers.Adam(lr=3.1415926e-4)

lossMean = metrics.Mean()
accuracy = metrics.Accuracy()

history_acc = 0
for epoch in range(20):

    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            logits = model(x)
            y = tf.one_hot(y, depth=classes)
            loss = tf.losses.categorical_crossentropy(y, logits, from_logits=True)
            loss = tf.reduce_mean(loss)
            lossMean.update_state(loss)
        gradient = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))

        if step % 10 == 0:
            print(epoch, step, "train loss:", lossMean.result().numpy())
            lossMean.reset_states()

    accuracy.reset_states()
    for (x, y) in val_db:
        logits = model(x)
        pro = tf.nn.softmax(logits, axis=1)
        pro = tf.cast(tf.argmax(pro, axis=1), dtype=tf.int32)
        accuracy.update_state(y, pro)

    print(epoch, "test correct:", accuracy.result().numpy())
