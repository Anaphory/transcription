#!python

import matplotlib.pyplot as plt
from segments import Tokenizer
from pathlib import Path
import numpy
from audio import x_play_file
import tensorflow as tf
from phonetic_features import N_FEATURES, feature_vector_of_sound, xsampa, features
from dataset import SpectrogramFeaturesSequence

tokenizer = Tokenizer()


def sigmoid(x, scale=10):
    return 1/(1+tf.exp(-x / scale))

if True:
        length_coefficients = tf.Variable(
            numpy.random.random((N_FEATURES, 1)),
            name="length_coefficients",
            dtype=numpy.float64)
        length_bias = tf.Variable(
            30,
            name="length_bias",
            dtype=numpy.float64)

        _N = tf.placeholder(tf.int64, name="N")
        _event_features = tf.placeholder(
            tf.float64, tf.TensorShape((None, N_FEATURES)),
            name="event_features")
        _target = tf.placeholder(
            tf.float64, tf.TensorShape((None, N_FEATURES)),
            name="real_feature_time_series")
        _pause = tf.Variable(
            [10.],
            tf.TensorShape((1)),
            dtype=tf.float64,
            name="pause_after")

        lengths = tf.concat((
            tf.matmul(
                _event_features,
                length_coefficients)[..., 0] + length_bias,
            _pause),
                            axis=0)
        raw_switches_right = tf.cumsum(lengths)
        raw_switches_left = tf.cumsum(lengths, exclusive=True)
        last_switch = raw_switches_right[..., -1]

        ranges = tf.reshape(tf.range(
            -tf.to_double(_N) + 0.5,
            0,
            dtype=tf.float64),
                            (-1, 1),
                            name="ranges")

        scale = tf.Variable(
            10,
            dtype=tf.float64, name="wideness")
        down = sigmoid(ranges - raw_switches_right + last_switch)
        up = sigmoid(ranges - raw_switches_left + last_switch)

        dir = tf.matmul(up - down, tf.concat(
            (_event_features,
             tf.zeros((1, N_FEATURES), dtype=tf.float64)), axis=0))

        c_ent = (_target - dir)**2
        err = tf.reduce_mean(c_ent)

        # err = tf.Print(err, [c_ent], summarize=10)

        opt = tf.train.AdagradOptimizer(100.)

        grad = opt.compute_gradients(
            err, var_list=[length_bias, length_coefficients, _pause])

        filtered_grad = [
            (tf.where(tf.is_nan(grad[0][0]),
                      tf.zeros_like(grad[0][0]),
                      grad[0][0]),
             grad[0][1]),
            (tf.where(tf.is_nan(grad[1][0]),
                      tf.zeros_like(grad[1][0]),
                      grad[1][0]),
             grad[1][1]),
            (tf.where(tf.is_nan(grad[2][0]),
                      tf.zeros_like(grad[2][0]),
                      grad[2][0]),
             grad[2][1])]


        apply = opt.apply_gradients(filtered_grad)

pauses_after = {}

for audio in (Path().absolute().parent / "data").glob("**/*i*.wav"):
    try:
        transcription = audio.with_suffix(".txt").open().read().strip()
        transcription = "_" + transcription
        spectrogram = numpy.load(audio.with_suffix(".npy").open("rb"))
        segments = tokenizer(transcription, ipa=True).split()
        time_series = SpectrogramFeaturesSequence.features_from_textgrid(
            audio, len(spectrogram))
        print(transcription)

        # plt.subplot(2, 1, 1)
        # plt.plot(numpy.exp(spectrogram).sum(1))
        # plt.xlim((0, len(spectrogram)))
        # plt.subplot(2, 1, 2)
        # plt.imshow(spectrogram.T, aspect="auto", origin="bottom")
        # x_play_file(audio)
        # plt.show()

        pauses_after[audio] = 1.
    except FileNotFoundError:
        continue

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    stats = []
    for audio in (Path().absolute().parent / "data").glob("**/*i*.wav"):
        try:
            transcription = audio.with_suffix(".txt").open().read().strip()
            # transcription = "_{0}a_a{0}a".format(transcription)
            transcription = "_" + transcription
            print(transcription)
            spectrogram = numpy.load(audio.with_suffix(".npy").open("rb"))
            segments = tokenizer(transcription, ipa=True).split()
            time_series = SpectrogramFeaturesSequence.features_from_textgrid(
                audio, len(spectrogram))
        except FileNotFoundError:
            print(">>", audio)
            continue

        N = len(time_series)

        event_features = numpy.array([
            feature_vector_of_sound(segment)
            for segment in segments])

        for e in event_features:
            print([f for f, p in zip(features, e)
                    if p])

        for i in range(6000):
            sess.run(_pause.assign([pauses_after[audio]]))
            sess.run(scale.assign(0.9999 ** i * 10))

            pause, pred, target, l, e0 = sess.run(
                [_pause, dir, _target, err, apply],
                feed_dict={
                    _N: N,
                    _event_features: event_features,
                    _target: time_series})

            print(l)
            print(pauses_after[audio])

        pauses_after[audio] = pause[0]

        print(sess.run(length_coefficients).reshape(-1))
        print(sess.run(length_bias))

        plt.subplot(3, 1, 1)
        plt.imshow(pred.T, aspect="auto", origin="bottom")
        plt.subplot(3, 1, 2)
        plt.imshow(target.T, aspect="auto", origin="bottom")
        plt.subplot(3, 1, 3)
        plt.imshow(spectrogram.T, aspect="auto", origin="bottom")
        plt.show()

