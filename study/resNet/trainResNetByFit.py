# use tensorflow-2.0
import os

import tensorflow as tf
from resNet import ResNet
from tensorflow.keras import datasets, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)


def preprocess(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1.
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=classes)
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

model.compile(optimizer=optimizer,
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_db, epochs=20, validation_data=val_db, validation_steps=1)

model.evaluate(val_db)
