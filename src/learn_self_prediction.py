#!/usr/bin/env python

import tensorflow as tf

import sys

import audio
from dataset import audio_data, read_audio, features_data
from spectrogram_to_sound import stft, griffin_lim
import model

import matplotlib.pyplot as plt
from build_signal import signal_from_norm_stft
import soundfile as sf
from phonetic_features import N_FEATURES

features_batch = features_data.padded_batch(2, padded_shapes=features_data.output_shapes)

audio_iterator = tf.data.Iterator.from_structure(features_batch.output_types,
                                                 features_batch.output_shapes)

audio_batch = audio_data.padded_batch(5, padded_shapes=features_data.output_shapes)

audio_init_op = audio_iterator.make_initializer(audio_batch)
features_init_op = audio_iterator.make_initializer(features_batch)

audio_for_features, features = audio_iterator.get_next(name='features_data')

magnitude_spectrograms = tf.abs(stft(audio_for_features))
output, loss, ahead, behind = model.lstm_network(magnitude_spectrograms, n_hidden=129)
sound_ahead = griffin_lim(ahead)
sound_behind = griffin_lim(behind)

features_output = model.feature_network(output)
feature_loss = tf.reduce_mean(tf.abs(
    tf.cast(features, tf.float32) - features_output))


audio_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
features_op = tf.train.AdamOptimizer(1e-3).minimize(feature_loss)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(20):
        print("\n", i)
        sess.run(audio_init_op)

        while False:
            try:
                l, _ = sess.run((loss, audio_op))
                print(l, end=" ")
                # for s, s_a, s_b in zip(ss, ss_a, ss_b):
                #     audio.scipy_play(256 * (s - s.min()) / (s.max() - s.min()))
                #     audio.scipy_play(256 * (s_a - s_a.min()) / (s_a.max() - s_a.min()))
                #     audio.scipy_play(256 * (s_b - s_b.min()) / (s_b.max() - s_b.min()))
            except tf.errors.OutOfRangeError:
                break
            sys.stdout.flush()

        print("\n", i)
        sess.run(features_init_op)

        while True:
            try:
                l, _ = sess.run((feature_loss, features_op))
                print(l, end=" ")
                # for s, s_a, s_b in zip(ss, ss_a, ss_b):
                #     audio.scipy_play(256 * (s - s.min()) / (s.max() - s.min()))
                #     audio.scipy_play(256 * (s_a - s_a.min()) / (s_a.max() - s_a.min()))
                #     audio.scipy_play(256 * (s_b - s_b.min()) / (s_b.max() - s_b.min()))
            except tf.errors.OutOfRangeError:
                break
            sys.stdout.flush()

