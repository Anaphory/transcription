#!/usr/bin/env

"""Various sound datasets"""

import numpy
import bisect
from pathlib import Path

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


SEGMENTS = ['', '#H', '*', '2:', '6', '9', '@', 'ASpause', 'C', 'E', 'E:',
            'H#', 'I', 'I6', 'N', 'O', 'Q', 'S', 'U', 'Y', 'a', 'a:', 'b', 'd',
            'e:', 'f', 'g', 'h', 'i:', 'j', 'k', 'l', 'm', 'n', 'o:', 'p', 'r',
            's', 't', 'u:', 'v', 'x', 'y:', 'z']


class TimeAlignmentSequence(Sequence):
    def __init__(self, batch_size=10, files=None):
        if files is None:
            files = DATA_PATH.glob("*.TextGrid")
        self.batch_size = batch_size
        sizes = []
        self.files = []
        for file in files:
            try:
                sg = numpy.load(file.with_suffix(".npy").open("rb"))
            except (OSError, FileNotFoundError):
                from audio_prep import read_audio
                sg = read_audio(file.with_suffix(".wav"), 'wav')
                numpy.save(file.with_suffix(".npy").open("wb"), sg)
            i = bisect.bisect(sizes, len(sg))
            sizes.insert(i, len(sg))
            self.files.insert(i, file)
        self.max_len = sizes[-1]

        self.index_correction = list(range(len(self)))
        numpy.random.shuffle(self.index_correction)

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
                 self.max_len),
                dtype=int)
            label_lengths = numpy.zeros(
                len(spectrograms),
                dtype=int)

            for i, file in enumerate(files):
                ls = self.labels_from_textgrid(
                        file)
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
    def labels_from_textgrid(file):
        with file.with_suffix(".TextGrid").open() as tr:
            textgrid = read_textgrid.TextGrid(tr.read())

        phonetics = textgrid.tiers[0]

        windows_per_second = 1000 / hparams["frame_shift_ms"]

        feature_matrix = []
        for start, end, segment in phonetics.simple_transcript:
            feature_matrix.append(SEGMENTS.index(segment))
        return feature_matrix
