# use tensorflow-2.0
import os

import tensorflow as tf
from resNet import ResNet
from tensorflow.keras import datasets, metrics, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)


def preprocess(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


classes = 10
## correct:92.59%
(x, y), (val_x, val_y) = datasets.fashion_mnist.load_data()
## correct:99.11%
# (x, y), (val_x, val_y) = datasets.mnist.load_data()
input_shape = (None, 28, 28, 1)
x = tf.reshape(x, (-1, 28, 28, 1))
val_x = tf.reshape(val_x, (-1, 28, 28, 1))


# classes = 10
## correct:79.23%
# (x, y), (val_x, val_y) = datasets.cifar10.load_data()

# classes = 100
## correct:45.18%
# (x, y), (val_x, val_y) = datasets.cifar100.load_data()

# input_shape = (None, 32, 32, 3)
# y = tf.squeeze(y, axis=1)
# val_y = tf.squeeze(val_y, axis=1)
# y = tf.squeeze(y, axis=1)
# val_y = tf.squeeze(val_y, axis=1)


train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(10000).batch(250)

val_db = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_db = val_db.map(preprocess).batch(250)

model = ResNet(layer_dims=[2, 2, 2, 2], classes_num=classes)
model.build(input_shape=input_shape)
model.summary()

lossMean = metrics.Mean()
accuracy = metrics.Accuracy()

optimizer = optimizers.Adam(lr=3.1415926e-4)

# model.load_weights("checkpoint/testResNet")

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
    # if accuracy.result().numpy() > history_acc:
    #     history_acc = accuracy.result().numpy()
    #     model.save_weights("checkpoint/testResNet")
