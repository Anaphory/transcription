#!/usr/bin/env python

from dataset import audio_dataset
from model import lstm_network
import tensorflow as tf

import matplotlib.pyplot as plt
from build_signal import signal_from_norm_stft
import soundfile as sf

dataset = audio_dataset.padded_batch(5, padded_shapes=[None])

iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                           dataset.output_shapes)

signals = iterator.get_next()

magnitude_spectrograms = tf.abs(tf.contrib.signal.stft(
            signals,
            frame_length=1024, # The normal sample rate is 20000Hz, so this is 20 ms
            # Periodic Hann is the default window
            frame_step=128)) # 6.4 ms

output, loss, ahead, behind = lstm_network(magnitude_spectrograms)

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

init_op = tf.global_variables_initializer()

training_iterator = iterator.make_initializer(dataset)
validation_iterator = iterator.make_initializer(dataset)

with tf.Session() as sess:
    sess.run(init_op)

    sess.run(training_iterator)
    while True:
        try:
            sess.run(train_op)
        except tf.errors.OutOfRangeError:
            break

    sess.run(validation_iterator)
    i = 0
    while True:
        try:
            for s, m, a, b in zip(*sess.run([signals, magnitude_spectrograms, ahead, behind])):
                print(i)
                plt.subplot(4, 1, 1)
                plt.plot(s)
                sf.write("sound_original_{:d}.wav".format(i), s, 20000)
                plt.subplot(4, 1, 2)
                m = signal_from_norm_stft(m)
                plt.plot(m)
                sf.write("sound_spectrogram_{:d}.wav".format(i), m, 20000)
                plt.subplot(4, 1, 3)
                a = signal_from_norm_stft(a)
                plt.plot(a)
                sf.write("sound_ahead_{:d}.wav".format(i), a, 20000)
                plt.subplot(4, 1, 4)
                b = signal_from_norm_stft(b)
                plt.plot(b)
                sf.write("sound_behind_{:d}.wav".format(i), b, 20000)
                i += 1
        except tf.errors.OutOfRangeError:
            break





