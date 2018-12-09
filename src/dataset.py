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

SEGMENTS = ['*', '2:', '6', '9', '@', 'C', 'E', 'E:',
            'I', 'I6', 'N', 'O', 'Q', 'S', 'U', 'Y', 'a', 'a:', 'b', 'd',
            'e:', 'f', 'g', 'h', 'i:', 'j', 'k', 'l', 'm', 'n', 'o:', 'p', 'r',
            's', 't', 'u:', 'v', 'x', 'y:', 'z']


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
            feature_values = numpy.zeros(
                    (len(spectrograms),
                     len(spectrograms[-1]),
                     len(SEGMENTS) + 1))

            for i, file in enumerate(files):
                feature_values[i] = self.features_from_textgrid(
                        file, spectrograms.shape[1])
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

        feature_matrix = numpy.zeros(
            (spectrogram_length,
             len(SEGMENTS) + 1),
            dtype=bool)
        for start, end, segment in phonetics.simple_transcript:
            if segment not in SEGMENTS:
                segment = "*"
            start = float(start)
            end = float(end)
            feature_matrix[
                int(start * windows_per_second):int(end * windows_per_second),
                SEGMENTS.index(segment)] = 1
        return feature_matrix


class ToStringSequence(TimeAlignmentSequence):
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
            spectrogram_lengths = numpy.zeros(
                len(sgs),
                dtype=int)
            for i, sg in enumerate(sgs):
                spectrograms[i][:len(sg)] = sg
                spectrogram_lengths[i] = len(sg)

            files = self.files[index * self.batch_size:
                        	   (index + 1) * self.batch_size]
            labels = numpy.zeros(
                (len(spectrograms),
                 hparams["max_string_length"]),
                dtype=int)
            label_lengths = numpy.zeros(
                len(spectrograms),
                dtype=int)

            for i, file in enumerate(files):
                ls = self.labels_from_textgrid(
                        file, spectrograms.shape[1])
                label_lengths[i] = len(ls)
                labels[i][:len(ls)] = ls
        except Exception as e:
            # Keras eats all errors, make sure to at least see them in the console
            print(e, end="\n\n")

        return ([spectrograms,
                 labels,
                 spectrogram_lengths,
                 label_lengths],
                [numpy.zeros(len(spectrograms))])

    @staticmethod
    def labels_from_textgrid(file, spectrogram_length=100):
        with file.with_suffix(".TextGrid").open() as tr:
            textgrid = read_textgrid.TextGrid(tr.read())

        phonetics = textgrid.tiers[0]

        windows_per_second = 1000 / hparams["frame_shift_ms"]

        feature_matrix = []
        for start, end, segment in phonetics.simple_transcript:
            if float(start) < spectrogram_length / windows_per_second:
                feature_matrix.append(SEGMENTS.index(segment))
        return feature_matrix


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
            labels = -numpy.ones(
                (len(data),
                 hparams["max_string_length"]),
                dtype=int)
            label_lengths = numpy.zeros(
                (len(data), 1),
                dtype=int)


            for i, (file, slice) in enumerate(data):
                sg = numpy.load(file.with_suffix(".npy").open("rb"))
                s = len(sg[slice])
                spectrograms[i][:s] = sg[slice]
                spectrogram_lengths[i] = s

                ls = [k for k, g in itertools.groupby(
                    numpy.argmax(self.features_from_textgrid(
                        file, len(sg)), 1)[slice])]
                if not ls:
                    raise ValueError
                label_lengths[i] = len(ls)
                labels[i][:len(ls)] = ls

        except Exception as e:
            # Keras eats all errors, make sure to at least see them in the console
            import traceback
            traceback.print_exc()
            import pdb; pdb.set_trace()

        return ([spectrograms,
                 labels,
                 spectrogram_lengths,
                 label_lengths],
                [numpy.zeros(len(spectrograms))])
