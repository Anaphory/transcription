#!/usr/bin/env python

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
# Obtained under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# and modified.
#
# Copyright 2018 Gereon Kaiping.

import itertools
from pathlib import Path

from keras.utils import Sequence

import bisect
import numpy
import tensorflow as tf

from phonetic_features import N_FEATURES, feature_vector_of_sound, xsampa
import read_textgrid
from hparams import hparams as p
prediction_length_windows = -(-p["prediction_time_at_least_ms"] //
                              p["frame_shift_ms"])
# Negative number magic to round to the next int

from spectrogram_to_sound import stft
from keras.preprocessing.sequence import pad_sequences

this = Path(__file__)
DATA_PATH = this.parent.parent / "data"


def list_wavfiles():
    for file in itertools.chain(DATA_PATH.glob("**/*.wav")):
        yield file.absolute()


def wavfile_with_textgrid(files=list(list_wavfiles()), shuffle=True):
    if shuffle:
        numpy.random.shuffle(files)
    for file in files:
        try:
            with file.with_suffix(".TextGrid").open() as tr:
                textgrid = tr.read()
        except UnicodeDecodeError:
            tr.close()
            with file.with_suffix(".TextGrid").open(encoding="utf-16") as tr:
                textgrid = tr.read()
        except FileNotFoundError:
            textgrid = None
            feature_matrix = None

        path, format = str(file), file.suffix[1:]
        audio = read_audio(path, format)

        if textgrid:
            textgrid = read_textgrid.TextGrid(textgrid)

            phonetics = textgrid.tiers[0]
            if phonetics.nameid == "Phonetic":
                # Assume XSAMPA
                transform = xsampa
            elif phonetics.nameid == "PhoneticIPA":
                transform = None
            elif phonetics.nameid == "MAU": # WebMAUS output
                transform = None
            else:
                raise ValueError("Unexpected tier found in file {:}: {:}".format(
                    file, phonetics.nameid))

            windows_per_second = 1000 / p["frame_shift_ms"]

            feature_matrix = numpy.zeros(
                (int(float(phonetics.simple_transcript[-1][1]) *
                     windows_per_second -
                     p["frame_length_ms"] / p["frame_shift_ms"]) +
                 1,
                N_FEATURES),
                dtype=bool)
            for start, end, segment in phonetics.simple_transcript:
                start = float(start)
                end = float(end)
                if transform:
                    segment = transform(segment)
                window_features = feature_vector_of_sound(segment)
                for window in range(int(start * windows_per_second),
                                    int(end * windows_per_second)):
                    try:
                        feature_matrix[window] = window_features
                    except IndexError:
                        continue

        yield audio, feature_matrix


def audio_path_and_type(files=list(list_wavfiles())):
    for path in files:
        yield str(path), path.suffix[1:]


g = tf.Graph()
with g.as_default():
    waveform = tf.squeeze(
        tf.contrib.ffmpeg.decode_audio(
            tf.read_file(tf.placeholder(tf.string, name="filename")),
            file_format=tf.placeholder(tf.string, name="format"),
            samples_per_second=p["sample_rate"],
            channel_count=1))

    log_mag_spectrogram = tf.log(tf.abs(stft(waveform) + 1e-8))


def read_audio(filename, format):
    with tf.Session(graph=g) as sess:
        return sess.run(waveform, feed_dict={
            'filename:0': str(filename),
            'format:0': format})


def spectrogram(wave):
    with tf.Session(graph=g) as sess:
        return sess.run(log_mag_spectrogram, feed_dict={
            waveform: wave})


class AudiofileSequence(Sequence):
    def __init__(self, batch_size=10, files=None):
        if files is None:
            files = list_wavfiles()
        self.batch_size = batch_size
        sizes = []
        self.files = []
        for file in files:
            try:
                sg = numpy.load(file.with_suffix(".npy").open("rb"))
            except (OSError, FileNotFoundError):
                waveform = read_audio(file, 'wav')
                sg = spectrogram(waveform)
                numpy.save(file.with_suffix(".npy").open("wb"), sg)
            i = bisect.bisect(sizes, len(sg))
            sizes.insert(i, len(sg))
            self.files.insert(i, file)

        self.index_correction = list(range(len(self)))
        numpy.random.shuffle(self.index_correction)

    def __len__(self):
        return -(-len(self.files) // self.batch_size)

    def __getitem__(self, index):
        spectrograms = [
            numpy.load(file.with_suffix(".npy").open("rb"))
            for file in self.files[index * self.batch_size: (index + 1) * self.batch_size]]
        return pad_sequences(spectrograms,
                             dtype=numpy.float32,
                             value=numpy.log(1e-8))


class ShiftedSpectrogramSequence(AudiofileSequence):
    def __getitem__(self, index):
        raw = super().__getitem__(index)
        return (
            raw[:, prediction_length_windows:-prediction_length_windows],
            [raw[:, :-2 * prediction_length_windows],
             raw[:, 2 * prediction_length_windows:]])


class SpectrogramFeaturesSequence(AudiofileSequence):
    def __init__(self, batch_size=10, files=list_wavfiles()):
        files = [file
                 for file in files
                 if file.with_suffix(".TextGrid").exists()]
        super().__init__(batch_size=batch_size, files=files)

    def __getitem__(self, index):
        try:
            spectrograms = super().__getitem__(index)
            files = self.files[index * self.batch_size:
                        	   (index + 1) * self.batch_size]
            feature_values = [
                numpy.zeros(
                    (spectrograms.shape[0], spectrograms.shape[1], 2))
                for feature_id in range(N_FEATURES)]
            for i, file in enumerate(files):
                for f, feature in enumerate(self.features_from_textgrid(
                        file, spectrograms.shape[1]).T):
                    feature_values[f][i, :, 0] = 1 - feature
                    feature_values[f][i, :, 1] = feature
        except Exception as e:
            # Keras eats all errors, make sure to at least see them in the console
            print(e, end="\n\n")

        return (spectrograms, feature_values)

    @staticmethod
    def features_from_textgrid(file, spectrogram_length):
        with file.with_suffix(".TextGrid").open() as tr:
            textgrid = read_textgrid.TextGrid(tr.read())

        phonetics = textgrid.tiers[0]
        if phonetics.nameid == "Phonetic":
            # Assume XSAMPA
            transform = xsampa
        elif phonetics.nameid == "PhoneticIPA":
            transform = None
        elif phonetics.nameid == "MAU": # WebMAUS output
            transform = None
        else:
            raise ValueError("Unexpected first tier found in file {:}: {:}".format(
                file, phonetics.nameid))

        windows_per_second = 1000 / p["frame_shift_ms"]

        feature_matrix = numpy.zeros(
            (spectrogram_length, N_FEATURES),
            dtype=bool)
        for start, end, segment in phonetics.simple_transcript:
            start = float(start)
            end = float(end)
            if transform:
                segment = transform(segment)
            window_features = feature_vector_of_sound(segment)
            for window in range(int(start * windows_per_second),
                                int(end * windows_per_second)):
                try:
                    feature_matrix[window] = window_features
                except IndexError:
                    continue
        return feature_matrix

