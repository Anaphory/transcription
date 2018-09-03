#!python

from segments import Tokenizer
from pathlib import Path
import numpy
import tensorflow as tf
from phonetic_features import N_FEATURES, feature_vector_of_sound, xsampa
from dataset import SpectrogramFeaturesSequence

tokenizer = Tokenizer()


def sigmoid(x):
    return 1/(1+numpy.exp(-8*x))


class ErrorCalculator():
    def __init__(self, bias):
        self.length_coefficients = numpy.random.random(N_FEATURES)
        self.length_bias = bias

    def stretch(self, N, event_features):
        lengths = numpy.dot(
            event_features,
            self.length_coefficients) + self.length_bias
        raw_switches = numpy.hstack(([0], numpy.cumsum(lengths)))
        switches = N + raw_switches - raw_switches[-1]

        translation = numpy.zeros((N, len(lengths)))
        for i, (begin, end) in enumerate(zip(switches, switches[1:])):
            r = numpy.arange(len(translation))
            translation[:, i] = sigmoid(r + 0.5 - begin) - sigmoid(r + 0.5 - end)
        return numpy.dot(translation, event_features)

    def error(self, time_series, time_series_estimated):
        return ((time_series_estimated - time_series) ** 2).mean()

err = ErrorCalculator(20)

stats = []
for i in range(2000000):
    print(err.length_coefficients)
    print(err.length_bias)
    for audio in (Path().absolute().parent / "data").glob("**/*.wav"):
        try:
            transcription = audio.with_suffix(".txt").open().read().strip()
            transcription = "_" + transcription + "_"
            spectrogram = numpy.load(audio.with_suffix(".npy").open("rb"))
            while numpy.exp(spectrogram[-1]).sum() < 20:
                spectrogram = spectrogram[:-1]
            while numpy.exp(spectrogram[0]).sum() < 20:
                spectrogram = spectrogram[1:]
            segments = tokenizer(transcription, ipa=True).split()
            time_series = SpectrogramFeaturesSequence.features_from_textgrid(
                audio, len(spectrogram))
        except FileNotFoundError:
            continue

        N = len(time_series)

        stats.append(len(spectrogram) / len(segments))
        event_features = numpy.array([
            feature_vector_of_sound(segment)
            for segment in segments])
        e0 = err.stretch(N, event_features)
        change = numpy.random.normal(0, 0.1)
        index = numpy.random.randint(-1, len(err.length_coefficients))
        if index == -1:
            err.length_bias += change
        else:
            err.length_coefficients[index] += change
        e1 = err.stretch(N, event_features)
        if not (e1 < e0):
            if index == -1:
                err.length_bias -= change
            else:
                err.length_coefficients[index] -= change
