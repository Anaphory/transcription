#!/usr/bin/env python

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


