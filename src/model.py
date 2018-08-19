import tensorflow as tf

from hparams import hparams

def log_mel_cepstrogram(magnitude_spectrograms, num_mel_bins=32):
    """Convert a magnitude spectrogram (eg. from stft) into a log mel cepstrogram.

    return the mfccs tensor.

    """
    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
    lower_edge_hertz, upper_edge_hertz = 80.0, 7600.0
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, hparams.sample_rate, lower_edge_hertz, upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
    # Note: Shape inference for `tf.tensordot` does not currently handle this case.
    mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))
    log_offset = 1e-6
    log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)
    return tf.contrib.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)


def lstm_network(input_spectrum, batch_size=1, n_hidden=513,
                 time_length=10):
    """Generate a Tensorflow RNN calculation graph.

    Generate a calculation graph mapping a spectrogram of size (in the
    frequency axis) `spectrogram_size`, i.e. of shape [batch_size, sample_size,
    spectrogram_size] to two expected spectrograms of the same shape (one
    look-ahead, one look-behind), using an LSTM RNN.

    Return the input, expected-output, output and loss tensors.

    """
    with tf.name_scope('timeflies'):
        spectrogram_size = input_spectrum.shape[-1].value

        padded_spectrum = tf.pad(input_spectrum, [[0, 0], [time_length, time_length], [0, 0]])

        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(
            n_hidden, forget_bias=1.0, state_is_tuple=True)
        outputs, state = tf.nn.dynamic_rnn(
            lstm_bw_cell, padded_spectrum,
            dtype=tf.float32)
        lookahead_output = tf.contrib.layers.fully_connected(
            inputs=outputs, num_outputs=spectrogram_size)
        lookbehind_output = tf.contrib.layers.fully_connected(
            inputs=outputs, num_outputs=spectrogram_size)
        expected_lookahead_spectrum = tf.pad(input_spectrum, [[0, 0], [0, 2 * time_length], [0, 0]])
        expected_lookbehind_spectrum = tf.pad(input_spectrum, [[0, 0], [2 * time_length, 0], [0, 0]])

        log_offset = 1e-6

        ahead_deviation = tf.reduce_mean(tf.square(
            tf.log(lookahead_output + log_offset) -
            tf.log(expected_lookahead_spectrum + log_offset)))

        behind_deviation = tf.reduce_mean(tf.square(
            tf.log(lookbehind_output + log_offset) -
            tf.log(expected_lookbehind_spectrum + log_offset)))

        ahead_mel_deviation = tf.reduce_mean(tf.square(
            log_mel_cepstrogram(lookahead_output) -
            log_mel_cepstrogram(expected_lookahead_spectrum)))

        behind_mel_deviation = tf.reduce_mean(tf.square(
            log_mel_cepstrogram(lookbehind_output) -
            log_mel_cepstrogram(expected_lookbehind_spectrum)))

        loss = ((ahead_deviation + behind_deviation) * 100 +
                (ahead_mel_deviation + behind_mel_deviation) * 6)
    return outputs, loss, lookahead_output, lookbehind_output
