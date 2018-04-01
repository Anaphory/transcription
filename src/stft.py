#!/usr/bin/env python

from pathlib import Path
import tensorflow as tf
import soundfile as sf
import numpy
import matplotlib.pyplot as plt
import bisect

tf.reset_default_graph()

try:
    this = Path(__file__)
except NameError:
    this = Path("audio_rnn.py").absolute()
DATA_PATH = this.parent.parent / "data"

file = DATA_PATH / "De-Bett.ogg"


def stft_graph(frame_length=512, frame_step=512 // 4):
    """Generate a Tensorflow stft calculation graph.

    Return the input (signal) and output (spectrogram) nodes.
    """

    # A batch of float32 time-domain signals in the range [-1, 1] with
    # shape [batch_size, signal_length]. Both batch_size and signal_length
    # may be unknown.
    signal = tf.placeholder(tf.float32, [None, None])

    # `magnitude_spectrograms` is a [batch_size, ?, 129] tensor of
    # spectrograms. We would like to produce overlapping fixed-size
    # spectrogram patches; for example, for use in a situation where a
    # fixed size input is needed.
    magnitude_spectrograms = tf.abs(tf.contrib.signal.stft(
        signal,
        frame_length=frame_length,
        # Periodic Hann is the default window
        frame_step=frame_step))
    log_spec = tf.log(magnitude_spectrograms + tf.constant(1e-8))
    return signal, magnitude_spectrograms


batch_size = 1
spectrogram_size = 257
n_cell_dim = 129
with tf.name_scope('lstm'):
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(
        n_cell_dim, forget_bias=1.0, state_is_tuple=True)
    spectrum = tf.placeholder(tf.float32, [None, None, spectrogram_size])
    outputs, state = tf.nn.dynamic_rnn(
        lstm_bw_cell, spectrum,
        dtype=tf.float32)
    output = tf.contrib.layers.fully_connected(
        inputs=outputs, num_outputs=spectrogram_size)
    expected_spectrum = tf.placeholder(
        tf.float32, [None, None, spectrogram_size])
    loss = tf.reduce_mean(tf.square(output - expected_spectrum))


def plot(spectrograms, transcription):
    plt.title(transcription)
    for i, spectrogram in enumerate(spectrograms):
        plt.subplot(len(spectrograms), 1, i + 1)
        plt.imshow(spectrogram.T[::-1], aspect="auto",
                   extent=[0, len(spectrogram) / samplerate * 128,
                           0, 256 * samplerate])
    plt.show()


with tf.Session() as session:
    signal, magnitude_spectrograms = stft_graph()
    for i, file in enumerate(DATA_PATH.glob("*.ogg")):
        if i > 0:
            break
        transcription = file.with_suffix(".txt").open().read()
        data, samplerate = sf.read(file.open("rb"))

        if len(data.shape) > 1 and data.shape[1] == 2:
            data = data[0]

        spectrograms = session.run(
            magnitude_spectrograms,
            feed_dict={signal: [data]})
        for spectrogram in spectrograms:
            plot([spectrogram], transcription)

with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    x = session.run(
        output,
        feed_dict={spectrum: [spectrogram]})
    print(x)

optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    data = []
    transcriptions = []
    for i, file in enumerate(DATA_PATH.glob("*.ogg")):
        if i > 50:
            break
        transcriptions.append(file.with_suffix(".txt").open().read())
        blob, samplerate = sf.read(file.open("rb"))
        # We should re-sample if the sample rate is abnormal

        if len(blob.shape) > 1 and blob.shape[1] == 2:
            blob = blob[:, 0]

        spectrograms = session.run(
            magnitude_spectrograms,
            feed_dict={signal: [blob]})

        data.append(spectrograms)

    _loss = 1
    i = 0
    while _loss > 1e-3:
        for spectrogram in spectrograms:
            _loss, _spectrum, _expected_spectrum, _output, _ = session.run(
                [loss, spectrum, expected_spectrum, output, optimizer],
                feed_dict={
                    spectrum: spectrograms,
                    expected_spectrum: spectrograms
                })
        print(i, _loss)
        i += 1
