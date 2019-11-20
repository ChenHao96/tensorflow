# use tensorflow-2.0

import tensorflow as tf
from tensorflow.keras import datasets


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, (-1, 28 * 28))
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


_, (x_test, y_test) = datasets.fashion_mnist.load_data()

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(128)

model = tf.keras.models.load_model("fashion.h5")
model.evaluate(db_test)
