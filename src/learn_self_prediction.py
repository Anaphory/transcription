#!/usr/bin/env python

import audio
from dataset import audio_dataset
from spectrogram_to_sound import stft, griffin_lim
from model import lstm_network
import tensorflow as tf

dataset = audio_dataset.padded_batch(5, padded_shapes=[None])
dataset.shuffle(400)

iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                           dataset.output_shapes)
dataset_init_op = iterator.make_initializer(dataset)

signals = iterator.get_next()

magnitude_spectrograms = tf.abs(stft(signals))

output, loss, ahead, behind = lstm_network(magnitude_spectrograms)

sound_ahead = griffin_lim(ahead)
sound_behind = griffin_lim(behind)

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(20):
        print("\n", i)
        sess.run(dataset_init_op)

        while True:
            try:
                l, ss, ss_a, ss_b, _ = sess.run((loss, signals, sound_ahead, sound_behind, train_op))
                print(l, end=" ")
                # for s, s_a, s_b in zip(ss, ss_a, ss_b):
                #     audio.scipy_play(256 * (s - s.min()) / (s.max() - s.min()))
                #     audio.scipy_play(256 * (s_a - s_a.min()) / (s_a.max() - s_a.min()))
                #     audio.scipy_play(256 * (s_b - s_b.min()) / (s_b.max() - s_b.min()))
            except tf.errors.OutOfRangeError:
                break
