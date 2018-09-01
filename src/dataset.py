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

from phonetic_features import N_FEATURES, feature_vector_of_sound, xsampa
import read_textgrid
from hparams import hparams as p

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
        audio, _ = read_audio(path, format)

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


def read_audio(filename, format):
    waveform = tf.squeeze(
        tf.contrib.ffmpeg.decode_audio(
            tf.read_file(filename),
            file_format=format,
            samples_per_second=p["sample_rate"],
            channel_count=1))
    return waveform, tf.zeros((1, N_FEATURES), dtype=tf.bool)
