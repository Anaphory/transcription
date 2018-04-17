#!/usr/bin/env python

import numpy
from numpy import log, pad
from scipy.signal import resample

def normalize_sampling_rate(signal, original_rate, normal_rate=20000):
    """Resample `signal` to a fixed sampling rate.

    """
    if original_rate == normal_rate:
        return signal

    padded_length = 2 ** int(log(len(signal)) / log(2) + 1)

    padding = -len(signal) % padded_length

    padded_signal = pad(signal,
                        (padding // 2, padding - padding // 2),
                        'constant', constant_values=0)

    with_new_rate = resample(
        padded_signal,
        int(padded_length * normal_rate / original_rate + 0.5))

    unpadding = int(padding / 2 * normal_rate / original_rate + 0.6)
    return with_new_rate[unpadding: -unpadding]


def normalize_sampling_rate_windowed(signal, original_rate,
                                     normal_rate=44100, blocksize=None,
                                     overlap=True):
    """Resample `signal` to a fixed sampling rate.

    Chop `signal` into overlapping windows of length around 5 ms (or
    blocksize), resample each block and glue the results back together,
    averaging for the overlap.

    """
    if original_rate == normal_rate:
        return signal

    if blocksize is None:
        # The power of two closest to 5 ms
        blocksize = 2 ** int(log(0.005 * original_rate) / log(2) + 0.5)
        if blocksize < 2:
            raise ValueError("Signal has a resolution of less than 400 Hz, cannot resample")

    output_signal_length = 0
    output_signal_blocks = []
    for start in range(0, len(signal), blocksize):
        signal_block = signal[start: start + blocksize]
        length = int((start + len(signal_block)) *
                     normal_rate / original_rate + 0.5) - output_signal_length
        if not len(signal_block) or not length:
            continue
        output_signal_blocks.append(
            resample(numpy.hstack(
                [[0], signal_block, [0]]),
                        length + 2)[1: -1])
        output_signal_length += length

    if overlap:
        shifted = normalize_sampling_rate_windowed(
            signal[blocksize//2:], original_rate, normal_rate, blocksize, overlap=False)
        unshifted = numpy.hstack(output_signal_blocks)
        unshifted[-len(shifted):] = (unshifted[-len(shifted):] + shifted) / 2
        return unshifted
    else:
        return numpy.hstack(output_signal_blocks)
