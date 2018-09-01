"""Experiments on handling missing data in Keras"""

import keras
from keras.models import Model
from keras.optimizers import Adadelta
from keras.losses import mean_squared_error
from keras.layers import Input, Dense
from keras.callbacks import TerminateOnNaN

from keras import backend as K

import tensorflow as tf
import numpy

def loss_0_where_nan(loss_function, msg=""):
    def filtered_loss_function(y_true, y_pred):
        print(hidden_layer.get_weights())
        print(min_pred_layer.get_weights())
        print(max_min_pred_layer.get_weights())
        with_nans = loss_function(y_true, y_pred)
        nans = tf.is_nan(with_nans)
        filtered = tf.where(nans, tf.zeros_like(with_nans), with_nans)
        filtered = tf.Print(filtered,
                            [y_true, y_pred, nans, with_nans, filtered],
                            message=msg)
        return filtered
    return filtered_loss_function

input = Input(shape=(3,))

hidden_layer = Dense(2)
hidden = hidden_layer(input)
min_pred_layer = Dense(1)
min_pred = min_pred_layer(hidden)
max_min_pred_layer = Dense(1)
max_min_pred = max_min_pred_layer(hidden)

model = Model(inputs=[input],
              outputs=[min_pred, max_min_pred])
truncated_model = Model(inputs=[input],
                        outputs=[min_pred])

model.compile(
    optimizer=Adadelta(),
    loss=[loss_0_where_nan(mean_squared_error, "aux: "),
          loss_0_where_nan(mean_squared_error, "main: ")],
    loss_weights=[0.2, 1.0])

truncated_model.compile(
    optimizer=Adadelta(),
    loss=[loss_0_where_nan(mean_squared_error, "aux: ")])

def random_values(n, missing=False):
    for i in range(n):
        x = numpy.random.random(size=(2, 3))
        _min = numpy.minimum(x[..., 0], x[..., 1])
        if missing:
            _max_min = numpy.full((len(x), 1), numpy.nan)
        else:
            _max_min = numpy.maximum(_min, x[..., 2]).reshape((-1, 1))
        # print(x, numpy.array(_min).reshape((-1, 1)), numpy.array(_max_min), sep="\n", end="\n\n")

        if missing:
            yield x, numpy.array(_min).reshape((-1, 1))
        else:
            yield x, [numpy.array(_min).reshape((-1, 1)), numpy.array(_max_min)]

model.fit_generator(random_values(2, False),
                    steps_per_epoch=2,
                    verbose=False)
print("With missing")
print(hidden_layer.get_weights())
print(min_pred_layer.get_weights())
print(max_min_pred_layer.get_weights())
history = truncated_model.fit_generator(
    random_values(1, True),
    steps_per_epoch=1,
    verbose=False)
print(hidden_layer.get_weights())
print(min_pred_layer.get_weights())
print(max_min_pred_layer.get_weights())
print("Normal")
model.fit_generator(random_values(2, False),
                    steps_per_epoch=2,
                    verbose=False)

print(history.history)
