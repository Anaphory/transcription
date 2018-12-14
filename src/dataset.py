#!/usr/bin/env

"""Various sound datasets"""

from pathlib import Path
import numpy
import bisect
import itertools

import tensorflow as tf
from keras.utils import Sequence
from keras.preprocessing.sequence import pad_sequences

import read_textgrid

from hparams import hparams

try:
    this = Path(__file__).absolute()
except NameError:
    this = Path("__file__").absolute()
DATA_PATH = this.parent.parent / "data" / "selection"

from lingpy.sequence.sound_classes import sampa2uni
from pyclts import SoundClasses, TranscriptionSystem

systems = [SoundClasses("asjp"),
           SoundClasses("cv")]
features = [{c: i for i, c in enumerate(system.classes)}
            for system in systems]

def lookup_sampa(sampasymbol):
    try:
        return [feature[system[sampa2uni(sampasymbol)]]
                for feature, system in zip(features, systems)]
    except (AssertionError, KeyError):
        return [0 for feature in features]

class TimeAlignmentSequence(Sequence):
    def __init__(self, batch_size=10, files=None):
        if files is None:
            files = DATA_PATH.glob("*.TextGrid")
        self.batch_size = batch_size
        self.sizes = []
        self.files = []
        for file in files:
            try:
                sg = numpy.load(file.with_suffix(".npy").open("rb"))
            except (OSError, FileNotFoundError):
                from audio_prep import read_audio
                sg = read_audio(file.with_suffix(".wav"), 'wav')
                numpy.save(file.with_suffix(".npy").open("wb"), sg)
            i = bisect.bisect(self.sizes, len(sg))
            self.sizes.insert(i, len(sg))
            self.files.insert(i, file)

        self.index_correction = list(range(len(self)))
        self.index_correction.sort(key=lambda x: numpy.random.random())

    def __len__(self):
        return -(-len(self.files) // self.batch_size)

    def __getitem__(self, index):
        try:
            sgs = [
                numpy.load(file.with_suffix(".npy").open("rb"))
                for file in self.files[
                        index * self.batch_size: (index + 1) * self.batch_size]]
            spectrograms = numpy.zeros(
                    (len(sgs),
                     len(sgs[-1]),
                     len(sgs[-1][-1])))
            for i, sg in enumerate(sgs):
                spectrograms[i][:len(sg)] = sg

            files = self.files[index * self.batch_size:
                        	   (index + 1) * self.batch_size]
            feature_values = [numpy.zeros(
                    (len(spectrograms),
                     len(spectrograms[-1]),
                     len(f) + 1))
                              for f in features]

            for i, file in enumerate(files):
                 for feature_matrix, values in zip(
                         feature_values,
                         self.features_from_textgrid(
                             file, spectrograms.shape[1])):
                     feature_matrix[i][values] = 1
        except Exception as e:
            # Keras eats all errors, make sure to at least see them in the console
            print(e, end="\n\n")

        return (spectrograms, feature_values)

    @staticmethod
    def features_from_textgrid(file, spectrogram_length):
        with file.with_suffix(".TextGrid").open() as tr:
            textgrid = read_textgrid.TextGrid(tr.read())

        phonetics = textgrid.tiers[0]

        windows_per_second = 1000 / hparams["frame_shift_ms"]

        feature_strings = [
            numpy.zeros(spectrogram_length, dtype=int)
            for feature in features]
        for start, end, segment in phonetics.simple_transcript:
            start = float(start)
            end = float(end)
            for feature_string, value in zip(
                    feature_strings, lookup_sampa(segment)):
                feature_string[
                    int(start * windows_per_second):
                    int(end * windows_per_second)] = value
        return feature_strings


class ChoppedStringSequence(TimeAlignmentSequence):
    def __init__(self, chunk_size=20, **kwargs):
        self.chunks = []
        super().__init__(**kwargs)
        for file, size in zip(self.files, self.sizes):
            for i in range(0, size, chunk_size):
                chunk = slice(i, i+chunk_size)
                self.chunks.append((file, chunk))
        self.chunks.sort(key=lambda x: numpy.random.random())
        self.chunk_size = chunk_size

    def __len__(self):
        return -(-len(self.chunks) // self.batch_size)

    def __getitem__(self, index):
        try:
            data = self.chunks[
                index * self.batch_size: (index + 1) * self.batch_size]

            # Set up the output arrays.
            spectrograms = numpy.zeros(
                    (len(data),
                     self.chunk_size,
                     hparams["n_spectrogram"]))
            spectrogram_lengths = numpy.zeros(
                (len(data), 1),
                dtype=int)
            labels = [-numpy.ones(
                (len(data),
                 hparams["max_string_length"]),
                dtype=int)
                      for _ in features]
            label_lengths = numpy.zeros(
                (len(data), 1),
                dtype=int)


            for i, (file, slice) in enumerate(data):
                sg = numpy.load(file.with_suffix(".npy").open("rb"))
                s = len(sg[slice])
                spectrograms[i][:s] = 2 * (sg[slice]-sg.min())/(sg.max()-sg.min()) - 1
                if s <= hparams["chop"]:
                    s = hparams["chop"] + 1
                spectrogram_lengths[i] = s

                for f, feature in enumerate(self.features_from_textgrid(
                        file, len(sg))):
                    ls = [k for k, g in itertools.groupby(feature[slice])]
                    if not ls:
                        raise ValueError
                    label_lengths[i] = len(ls)
                    labels[f][i, :len(ls)] = ls

        except Exception as e:
            # Keras eats all errors, make sure to at least see them in the console
            import traceback
            traceback.print_exc()
            import pdb; pdb.set_trace()

        return ([spectrograms] +
                 labels + [
                 spectrogram_lengths,
                 label_lengths],
                [numpy.zeros(len(spectrograms))])
