#!/usr/bin/env python

import tensorflow as tf
from scipy.interpolate import interp1d
import numpy

def signal_from_norm_stft(spectrogram, step=128):
    time = spectrogram.shape[-2]
    interpolator = interp1d(
        numpy.arange(0, time * step, step),
        spectrogram,
        'linear',
        -2)
    interpolated = interpolator(numpy.arange(time * step - step + 1))
    phases = numpy.arange(0, time * step - step + 1) * 2 * numpy.pi
    frequencies = numpy.arange(spectrogram.shape[-1]) / (spectrogram.shape[-1] - 1)
    unfiltered = numpy.sin(numpy.outer(phases, frequencies))
    filtered = unfiltered * interpolated
    sum = filtered.sum(-1)
    return sum / sum.max()

def signal_from_norm_stft_(spectrogram, step=128):
    tfspectrogram = tf.placeholder(tf.complex64, spectrogram.shape)
    inv = tf.contrib.signal.inverse_stft(
        tfspectrogram,
        frame_length=1024, # The normal sample rate is 20000Hz, so this is 20 ms
        # Periodic Hann is the default window
        frame_step=step) # 6.4 ms
    import pdb; pdb.set_trace()
    with tf.Session() as sess:
        return sess.run(inv, feed_dict={tfspectrogram: spectrogram + 0.0j})
