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

import numpy
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import Dataset

from phonetic_features import N_FEATURES, feature_vector_of_sound, xsampa
import read_textgrid
import librosa

from hparams import hparams
try:
    this = Path(__file__)
except NameError:
    this = Path("dataset.py").absolute()
DATA_PATH = this.parent.parent / "data"


def list_wavfiles():
    for file in itertools.chain(DATA_PATH.glob("**/*.ogg"),
                                DATA_PATH.glob("**/*.wav")):
        yield file.absolute()


def wavfile_with_textgrid():
    for file in list_wavfiles():
        try:
            with file.with_suffix(".textgrid").open() as tr:
                textgrid = tr.read()
        except UnicodeDecodeError:
            tr.close()
            with file.with_suffix(".textgrid").open(encoding="utf-16") as tr:
                textgrid = tr.read()
        except FileNotFoundError:
            continue

        path, format = str(file), file.suffix[1:]
        audio = read_audio(path, format)

        textgrid = read_textgrid.TextGrid(textgrid)

        phonetics = textgrid.tiers[0]
        if phonetics.nameid == "Phonetic":
            # Assume XSAMPA
            transform = xsampa
        elif phonetics.nameid == "PhoneticIPA":
            transform = None
        else:
            raise ValueError("Unexpected tier found in file {:}: {:}".format(
                file, phonetics.nameid))

        windows_per_second = 1000 / hparams.frame_shift_ms

        feature_matrix = numpy.zeros(
            (int(phonetics.simple_transcript[-1][1] * windows_per_second + 1),
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
                feature_matrix[window] = window_features

        yield audio, feature_matrix


def audio_path_and_type():
    for path in list_wavfiles():
        yield str(path), path.suffix[1:]


def read_audio(filename, format):
    waveform = tf.squeeze(
        tf.contrib.ffmpeg.decode_audio(
            tf.read_file(filename),
            file_format=format,
            samples_per_second=hparams.sample_rate,
            channel_count=1))
    return waveform, tf.zeros((1, N_FEATURES), dtype=tf.bool)


audio_dataset = Dataset.from_generator(
    audio_path_and_type,
    (tf.string, tf.string),
    (tf.TensorShape(()), tf.TensorShape(())))


audio_data = audio_dataset.shuffle(1000).map(read_audio)

features_dataset = Dataset.from_generator(
    wavfile_with_textgrid,
    (tf.float32, tf.bool),
    (tf.TensorShape((None,)), tf.TensorShape((None, N_FEATURES))))

features_data = features_dataset.shuffle(1000)
