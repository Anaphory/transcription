#!/usr/bin/python

"""Find the vowels in an audio file"""

import soundfile as sf
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import scipy as sp

def specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    f, t, Sxx = spectrogram(audio, fs=sample_rate,
                            window='parzen',
                            nperseg=nperseg,
                            noverlap=noverlap,
                            detrend=False)
    return f, t, Sxx.astype(np.float32) + eps

def lpc(signal, order):
    """Compute the Linear Prediction Coefficients.

    Return the order + 1 LPC coefficients for the signal. c = lpc(x, k) will
    find the k+1 coefficients of a k order linear filter:

      xp[n] = -c[1] * x[n-2] - ... - c[k-1] * x[n-k-1]

    Such as the sum of the squared-error e[i] = xp[i] - x[i] is minimized.

    Parameters
    ----------
    signal: array_like
        input signal
    order : int
        LPC order (the output will have order + 1 items)"""

    order = int(order)

    if signal.ndim > 1:
        raise ValueError("Array of rank > 1 not supported yet")
    if order > signal.size:
        raise ValueError("Input signal must have a lenght >= lpc order")

    if order > 0:
        p = order + 1
        r = np.zeros(p, signal.dtype)
        # Number of non zero values in autocorrelation one needs for p LPC
        # coefficients
        nx = np.min([p, signal.size])
        x = np.correlate(signal, signal, 'full')
        r[:nx] = x[signal.size-1:signal.size+order]
        phi = np.dot(sp.linalg.inv(sp.linalg.toeplitz(r[:-1])), -r[1:])
        return np.concatenate(([1.], phi))
    else:
        return np.ones(1, dtype = signal.dtype)

data_path = Path("./").absolute().parent.parent / "data"
formants_by_vowel = {}
for file in data_path.glob("*vowel.ogg"):
    data, sample_rate = sf.read(str(file))
    if len(data.shape) == 2 and data.shape[1] == 2:
        data = data[:, 0]

    f, t, Sxx = specgram(data, sample_rate, window_size=5, step_size=2)
    #plt.imshow(np.log(Sxx)[::-1], extent=(t[0], t[-1], f[0], f[-1]), aspect="auto")

    energy = Sxx.sum(1)
    energy = (energy - energy.min()) / (energy.max() - energy.min())

    formants = f[
        [False] +
        [energy[i] > energy[i-1] and energy[i] > energy[i+1]
        for i in range(1, len(energy) -1)] +
        [False]]
    vowel = open(str(file)[:-4] + ".txt").read()
    #plt.title(vowel)
    #plt.show()
    formants_by_vowel.setdefault((formants[0], formants[1]), []).append(vowel)


if False:
    for formants, vowels in formants_by_vowel.items():
        plt.annotate(s=", ".join(vowels), xy=formants)
    plt.scatter(*zip(*formants_by_vowel.keys()))
    # plt.scatter(*zip(*upper_formants_by_vowel.keys()))
    plt.show()

ORDER = 8
for fs in range(5,122):
    x = np.arange(0, 90, 1/fs)
    signal = np.sin(x) + np.sin(x * 2) / 2 + np.sin(x * 3) / 3 + np.sin(x * 4) / 4
    A = lpc(signal, ORDER)
    roots = np.roots(A)
    roots = roots[roots.imag >= 0]
    angz = np.arctan2(roots.imag, roots.real)
    frequencies = angz * (fs / (2 * np.pi))
    bw = -1/2*(fs/(2*np.pi))*np.log(abs(roots))
    print(roots)

plt.plot(x, signal)

start = signal[:ORDER - 1]
est = []
for s in signal[ORDER:]:
    ...
