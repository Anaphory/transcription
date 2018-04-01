#!/usr/bin/env python

from pathlib import Path
import tensorflow as tf
import soundfile as sf
import numpy
import matplotlib.pyplot as plt
import bisect

try:
    this = Path(__file__)
except NameError:
    this = Path("audio_rnn.py").absolute()
DATA_PATH = this.parent.parent / "data"

tf.reset_default_graph()


def stft_graph(frame_length=256, frame_step=256 // 4):
    """Generate a Tensorflow stft calculation graph.

    Generate a spectrogram computation graph for the energy
    spectrogram of a signal. Return the input (signal) and output
    (spectrogram) nodes.

    """
    with tf.name_scope('spectrogram'):
        # A batch of float32 time-domain signals in the range [-1, 1] with
        # shape [batch_size, signal_length]. Both batch_size and signal_length
        # may be unknown.
        signal = tf.placeholder(tf.float32, [None, None])

        # `magnitude_spectrograms` is a [batch_size, ?, frame_length
        # // 2 + 1] tensor of spectrograms. We would like to produce
        # overlapping fixed-size spectrogram patches.
        magnitude_spectrograms = tf.abs(tf.contrib.signal.stft(
            signal,
            frame_length=frame_length,
            # Periodic Hann is the default window
            frame_step=frame_step))
        # Define tensor operations in case we want logarithmic
        # spectrograms instead of linear ones.
        log_spectrum = tf.log(magnitude_spectrograms + 1e-8)
        normalized = log_spectrum - tf.reduce_max(log_spectrum)
        return signal, magnitude_spectrograms


def lstm_network(batch_size=1, spectrogram_size=129, n_hidden=65):
    """Generate a Tensorflow RNN calculation graph.

    Generate a calculation graph mapping a spectrogram of size (in the
    frequency axis) `spectrogram_size`, i.e. of shape [batch_size,
    sample_size, spectrogram_size] to an expected spectrogram of the
    same shape, using an LSTM RNN.

    Return the input, expected-output, output and loss tensors.
    """
    with tf.name_scope('lstm'):
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(
            n_hidden, forget_bias=1.0, state_is_tuple=True)
        spectrum = tf.placeholder(tf.float32, [None, None, spectrogram_size])
        outputs, state = tf.nn.dynamic_rnn(
            lstm_bw_cell, spectrum,
            dtype=tf.float32)
        output = tf.contrib.layers.fully_connected(
            inputs=outputs, num_outputs=spectrogram_size)
        expected_spectrum = tf.placeholder(
            tf.float32, [None, None, spectrogram_size])
        loss = tf.reduce_mean(tf.square(output - expected_spectrum))
    return spectrum, expected_spectrum, output, loss


def plot(spectrograms, transcription, samplerate=44100,
         transform=lambda x: numpy.log(x + 1e-8), steps=256 // 4):
    """Plot one or more spectrograms in compatible axis.

    Parameters
    ----------
    spectrograms: sequence of 2d-arrays
      The spectrogram to be plotted
    transcription: string
      The figure's title

    Returns
    -------
    Nothing

    Side-Effects
    ------------
    Opens a window with the plots
    """
    plt.subplot(len(spectrograms), 1, 1)
    plt.title(transcription)
    vmin = transform(numpy.min(spectrograms))
    vmax = transform(numpy.max(spectrograms))
    for i, spectrogram in enumerate(spectrograms):
        spectrogram = transform(spectrogram)
        plt.subplot(len(spectrograms), 1, i + 1)
        plt.imshow(spectrogram.T[::-1], aspect="auto",
                   vmin=vmin, vmax=vmax,
                   extent=[0, len(spectrogram) / samplerate * steps,
                           0, samplerate / spectrogram.shape[1]])
    plt.show()

signal, magnitude_spectrograms = stft_graph()
spectrum, expected_spectrum, output, loss = lstm_network(n_hidden=33)

optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
session = tf.InteractiveSession()

session.run(tf.global_variables_initializer())

# Load the data
data = []
transcriptions = []

for i, file in enumerate(DATA_PATH.glob("*.ogg")):
    if len(data) > 50:
        break
    blob, samplerate = sf.read(file.open("rb"))
    # We should re-sample if the sample rate is abnormal
    if samplerate != 44100:
        print(file, "dropped: Incompatible sample rate", samplerate)
        continue
    transcriptions.append(file.with_suffix(".txt").open().read())

    if len(blob.shape) > 1 and blob.shape[1] == 2:
        blob = blob[:, 0]

    spectrograms = session.run(
        magnitude_spectrograms,
        feed_dict={signal: [blob]})

    data.append(spectrograms)

# Learn the network
epoch = 0
for epoch in range(512):
    for spectrogram in spectrograms:
        _loss, _spectrum, _expected_spectrum, _output, _ = session.run(
            [loss, spectrum, expected_spectrum, output, optimizer],
            feed_dict={
                spectrum: spectrograms,
                expected_spectrum: spectrograms
            })
    print(epoch, _loss)

# Plot the results
for spectrogram in data:
    _spectrum, _expected_spectrum, _output = session.run(
        [spectrum, expected_spectrum, output],
        feed_dict={
            spectrum: spectrogram,
            expected_spectrum: spectrogram})
    plot([_spectrum[0], _expected_spectrum[0], _output[0]], "")
