#!/usr/bin/env python

import collections
from pathlib import Path
import tensorflow as tf
import soundfile as sf
import numpy
import matplotlib.pyplot as plt
import bisect
import read_textgrid

try:
    this = Path(__file__)
except NameError:
    this = Path("audio_rnn.py").absolute()
DATA_PATH = this.parent.parent / "data"

tf.reset_default_graph()


def stft_graph(frame_length=1024, frame_step=128):
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
        normalized = magnitude_spectrograms / tf.reduce_max(log_spectrum)
        return signal, normalized


def lstm_network(batch_size=1, spectrogram_size=513, n_hidden=65,
                 return_hidden=False):
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
    if return_hidden:
        return spectrum, expected_spectrum, output, loss, outputs
    else:
        return spectrum, expected_spectrum, output, loss


def transcriber_network(hidden, n_classes):
    phonemes = tf.placeholder(tf.int16,
                              [None, None, n_classes])
    output = tf.contrib.layers.fully_connected(
        inputs=hidden, num_outputs=n_classes)
    softmax = tf.nn.softmax(output)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=softmax,
        labels=phonemes)
    return phonemes, output, tf.reduce_mean(loss)


def plot(spectrograms, transcription, samplerate=44100,
         transform=lambda x: numpy.log(x + 1e-8), steps=128):
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
                           0, samplerate / 2])
    plt.show()

sounds = collections.Counter()
for i, file in enumerate((DATA_PATH / "emu").glob("*.txt")):
    print(i, file)
    textgrid = read_textgrid.TextGrid(file.open().read())
    phonetics = textgrid.tiers[0]
    sounds += collections.Counter(
        p[2] for p in phonetics.simple_transcript)
n_sounds = len([x for x in sounds.values() if x>1])
labels = collections.OrderedDict()
for i, (sound, freq) in enumerate(
        sounds.most_common(n_sounds)):
    labels[sound] = i
labels["?"] = len(labels)


signal, magnitude_spectrograms = stft_graph()
spectrum, expected_spectrum, output, loss, hidden = lstm_network(
    n_hidden=33, return_hidden=True)
phonemes, inferred_classes, logit_loss = transcriber_network(hidden, len(labels))

optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
phon_optimizer = tf.train.AdamOptimizer(1e-3).minimize(logit_loss)
session = tf.InteractiveSession()

session.run(tf.global_variables_initializer())

# Load the data
data = []
transcriptions = []


def input_and_target(original, shift):
    """Generate input and target data from original.

    If shift == 0, input and target are identical (eg. for compression
    learning).  If shift > 0, target is ahead of input, so the network
    is taught to predict future samples.  If shift < 0, target is
    behind input, so the network is taught to remember past samples.

    """
    if shift == 0:
        return original, original
    elif shift > 0:
        original = numpy.asarray(original)
        shape = list(original.shape)
        shape[0] = shift
        padding = numpy.zeros(shape)
        input = numpy.concatenate((padding, original))
        target = numpy.concatenate((original, padding))
    else:
        original = numpy.asarray(original)
        shape = list(original.shape)
        shape[0] = -shift
        padding = numpy.zeros(shape)
        input = numpy.concatenate((original, padding))
        target = numpy.concatenate((padding, original))
    return input, target


for i, file in enumerate(DATA_PATH.glob("*.ogg")):
    if len(data) > 500:
        break
    blob, samplerate = sf.read(file.open("rb"))
    # We should re-sample if the sample rate is abnormal
    if samplerate != 44100 and samplerate != 44000:
        print(file, "dropped: Incompatible sample rate", samplerate)
        continue
    transcriptions.append(file.with_suffix(".txt").open().read())

    if len(blob.shape) > 1 and blob.shape[1] == 2:
        blob = blob[:, 0]

    spectrograms = session.run(
        magnitude_spectrograms,
        feed_dict={signal: [blob]})

    data.append(spectrograms)

shift = 1
inputs = []
targets = []
for spectrogram in data:
    input, target = input_and_target(spectrogram[0], shift)
    inputs.append(input)
    targets.append(target)
    
# Learn the network
for epoch in range(200):
    for input, target in zip(inputs, targets):
        _loss, _spectrum, _expected_spectrum, _output, _ = session.run(
            [loss, spectrum, expected_spectrum, output, optimizer],
            feed_dict={
                spectrum: [input],
                expected_spectrum: [target]})
    print(epoch, _loss)

# Plot the results
for input, target, trs in zip(inputs, targets, transcriptions):
    _spectrum, _expected_spectrum, _output = session.run(
        [spectrum, expected_spectrum, output],
        feed_dict={
            spectrum: [input],
            expected_spectrum: [target]})
    plot([_spectrum[0], _expected_spectrum[0], _output[0]], trs,
         transform=lambda x: x**0.5)
    break

ttrs = []
data = []
for i, textgrid_file in enumerate((DATA_PATH / "emu").glob("*.txt")):
    if len(data) > 100:
        break
    file = textgrid_file.with_suffix(".wav")
    blob, samplerate = sf.read(file.open("rb"))
    # We should re-sample if the sample rate is abnormal
    if samplerate != 20000 and samplerate != 16000:
        print(file, "dropped: Incompatible sample rate", samplerate)
        continue

    if len(blob.shape) > 1 and blob.shape[1] == 2:
        blob = blob[:, 0]

    spectrogram = session.run(
        magnitude_spectrograms,
        feed_dict={signal: [blob]})

    spectrogram = spectrogram[0]
    
    timed_transcription = numpy.zeros((len(spectrogram), len(labels)),
                                      dtype=numpy.int16)
    textgrid = read_textgrid.TextGrid(textgrid_file.open().read())
    for start, end, phone in textgrid.tiers[0].simple_transcript:
        int_start = round(float(start) / textgrid.t_time *
                          len(timed_transcription))
        int_end = round(float(end) / textgrid.t_time *
                        len(timed_transcription))
        if labels.get(phone):
            timed_transcription[int_start:int_end, labels[phone]] = 1
        
    ttrs.append(timed_transcription)
    data.append(spectrogram)

inputs = []
targets = []
for i, spectrogram in enumerate(data):
    input, target = input_and_target(spectrogram, shift)
    inputs.append(input)
    targets.append(target)
    ttrs[i], _ = input_and_target(ttrs[i], shift)

# Learn the network
for epoch in range(200):
    for input, target, ttr in zip(inputs, targets, ttrs):
        _loss, _logit_loss, _spectrum, _expected_spectrum, _output, _ = session.run(
            [loss, logit_loss, spectrum, expected_spectrum, output, phon_optimizer],
            feed_dict={
                spectrum: [input],
                expected_spectrum: [target],
                phonemes: [ttr],
                })
    print(epoch, _loss, _logit_loss)


unlabel = numpy.array(list(labels.keys()))
for input, target, ttr in zip(inputs, targets, ttrs):
    print()
    _inferred_classes = session.run(
        [inferred_classes],
        feed_dict={
            spectrum: [input],
            phonemes: [ttr]})

    prediction = unlabel[_inferred_classes[0][0].argmax(1)]
    original = unlabel[ttr.argmax(1)]
    print("".join("{:3s}".format(x) for x in original))
    print("".join("{:3s}".format(x) for x in prediction))
    print("".join(x for x, y in zip(original, original[1:]) if x!=y))
    print("".join(x for x, y in zip(prediction, prediction[1:]) if x!=y))
