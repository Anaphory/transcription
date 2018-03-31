#!/usr/bin/python

"""Find the vowels in an audio file"""

import soundfile as sf
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import scipy as sp

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

formants = [], []
data_path = Path("./").absolute().parent.parent / "data"
for file in data_path.glob("*vowel.ogg"):
    print(file)
    cdata, sample_rate = sf.read(str(file))

    ipa = open(str(file)[:-4]+".txt").read()
    
    if len(cdata.shape) == 2:
        if cdata.shape[1] == 2:
            cdata = cdata[:, 0]
        else:
            raise ValueError

    for i in range(0, len(cdata)-4000, 2000):
        data = cdata[i:i+4000]
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(data))/sample_rate, data)
        plt.legend(["Waveform"])
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

        plt.title(ipa)
        plt.subplot(2, 1, 2)
        # fft = sp.fft(data)[:200]
        # plt.plot(np.arange(len(fft)) * sample_rate / len(data), fft)

        ncoeff = 2 + sample_rate // 1000
        a = lpc(data, ncoeff)

        w, h = sp.signal.freqz(b=1, a=a)
        plt.plot(w / 2 / np.pi * sample_rate, 20*np.log(abs(h) + 1e-10)/np.log(10))

        r=np.roots(a) # find roots of polynomial a
        r=r[r.imag>0.01] # only look for roots >0Hz up to fs/2
        freqs = (np.arctan2(r.imag, r.real) / 2 / np.pi * sample_rate)
                                    # convert to Hz and sort
        bw = -1/2 * (sample_rate / 2 / np.pi) * np.log(abs(r))
        bw = bw[freqs > 150]
        freqs = freqs[freqs > 150]
        freqs = freqs[bw < 400]
        bw = bw [bw < 400]
        formants[0].append(ipa)
        formants[1].append(sorted(freqs))
        print(ipa)
        print(freqs)
        print(bw)
        plt.scatter(freqs, np.zeros_like(freqs))
        plt.xlim(0, 6000)
plt.show()

plt.scatter([f[0] for f in formants[1]],
            [f[1] for f in formants[1]])
for key, value in zip(*formants):
    plt.annotate(key+"  ", value[:2])
plt.show()

