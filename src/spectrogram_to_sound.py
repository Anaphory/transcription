import tensorflow as tf
from pathlib import Path
import numpy
import audio

from hparams import hparams

# Based on
# https://github.com/keithito/tacotron/blob/v0.2.0/util/audio.py,
# which is derived from
# https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb


def stft_parameters():
    n_fft = (hparams.num_freq - 1) * 2
    hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
    return n_fft, hop_length, win_length


def istft(stfts):
    n_fft, hop_length, win_length = stft_parameters()
    return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)


def stft(signals):
    n_fft, hop_length, win_length = stft_parameters()
    return tf.contrib.signal.stft(signals, win_length, hop_length,
                                  n_fft, pad_end=False)


def griffin_lim(spectrograms):
    '''Run the Griffin-Lim algorithm on the spectrograms.

    Transform the modulus short time fourier transform of a signal back to a
    signal by ‘inventing’ compatible phase information for each segment,
    iteratively improving the overlap throughout `hparams.griffin_lim_iters`
    iteration steps.

    Parameters
    ----------
    spectrograms: real tf.Tensor of shape (..., frames, stft_bins)

    Returns
    -------
    tf.Tensor of shape (..., samples)

    '''
    with tf.variable_scope('griffinlim'):
        # TensorFlow's stft and istft operate on a batch of
        # spectrograms; create batch of size 1
        S_complex = tf.identity(tf.cast(spectrograms, dtype=tf.complex64))
        y = istft(S_complex)
        for i in range(hparams.griffin_lim_iters):
            est = stft(y)
            angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
            y = istft(S_complex * angles)
        return y


if __name__ == "__main__":
    from dataset import audio_dataset
    dataset = audio_dataset.padded_batch(5, padded_shapes=[None])

    iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                            dataset.output_shapes)
    dataset_init_op = iterator.make_initializer(dataset)


    signals = iterator.get_next()

    magnitude_spectrograms = tf.abs(stft(signals))

    backwards = griffin_lim(magnitude_spectrograms)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(dataset_init_op)
        while True:
            try:
                befores, afters = sess.run((signals, backwards))
                for before, after in zip(befores, afters):
                    print(before.shape, after.shape)
                    audio.scipy_play(256 * (before - before.min()) / (before.max() - before.min()))
                    audio.scipy_play(256 * (after - after.min()) / (after.max() - after.min()))
            except tf.errors.OutOfRangeError:
                break
