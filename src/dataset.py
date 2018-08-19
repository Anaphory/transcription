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


def read_wavfile():
    for file in itertools.chain(DATA_PATH.glob("**/*.ogg"),
                                DATA_PATH.glob("**/*.wav")):
        waveform, samplerate = librosa.load(file, sr=hparams.sample_rate)
        if len(waveform.shape) > 1:
            waveform = waveform[:, 1]

        yield waveform


def read_wavfile_and_textgrid():
    for file in itertools.chain(DATA_PATH.glob("**/*.ogg"),
                                DATA_PATH.glob("**/*.wav")):
        try:
            try:
                with file.with_suffix(".textgrid").open() as tr:
                    textgrid = tr.read()
            except UnicodeDecodeError:
                tr.close()
                with file.with_suffix(".textgrid").open(encoding="utf-16") as tr:
                    textgrid = tr.read()
        except FileNotFoundError:
            continue

        textgrid = read_textgrid.TextGrid(textgrid)

        waveform, samplerate = librosa.load(file, sr=hparams.sample_rate)
        print(samplerate)
        if len(waveform.shape) > 1:
            waveform = waveform[:, 0]

        phonetics = textgrid.tiers[0]
        if phonetics.nameid == "Phonetic":
            # Assume XSAMPA
            transform = xsampa
        elif phonetics.nameid == "PhoneticIPA":
            transform = None
        else:
            raise ValueError("Unexpected tier found in file {:}: {:}".format(
                file, phonetics.nameid))
        segments = phonetics.simple_transcript

        length = len(waveform) / samplerate
        windows_per_second = 1000 / hparams.frame_shift_ms
        feature_matrix = numpy.zeros((int(length * windows_per_second),
                                      N_FEATURES))

        form = ""
        try:
            for start, end, segment in segments:
                start = float(start)
                end = float(end)
                if transform:
                    segment = transform(segment)
                form += segment
                window_features = feature_vector_of_sound(segment)
                for window in range(int(start * windows_per_second),
                                    int(end * windows_per_second)):
                    feature_matrix[window] = window_features
        except IndexError:
            print("Inconsistent textgrid for {:}, ignored.".format(file))
            continue

        if not feature_matrix.any():
            continue
        yield waveform, feature_matrix, form


def read_wavfile_and_annotation():
    for file in itertools.chain(DATA_PATH.glob("**/*.ogg"),
                                DATA_PATH.glob("**/*.wav")):
        transcription = None
        try:
            with file.with_suffix(".txt").open() as tr:
                transcription = tr.read()
                # Transcription needs to be segmented
        except FileNotFoundError:
            transcription = None
        try:
            try:
                with file.with_suffix(".textgrid").open() as tr:
                    textgrid = tr.read()
            except UnicodeDecodeError:
                tr.close()
                with file.with_suffix(".textgrid").open(encoding="utf-16") as tr:
                    textgrid = tr.read()
            textgrid = read_textgrid.TextGrid(textgrid)
            phonetics = textgrid.tiers[0]
            segments = phonetics.simple_transcript
            transcription = [ts.resolve_sound(clean(segment[2])) for segment in segments]
            if [t for t in transcription if type(t) == pyclts.models.UnknownSound]:
                print(file, " ".join(map(str, transcription)))
            segments = [feature_vector_of_sound(clean(segment[2]))
                        for segment in segments]
        except FileNotFoundError:
            segments = None

        waveform, samplerate = sf.read(file.open("rb"))
        if len(waveform.shape) > 1:
            waveform = waveform[:, 1]
        waveform = normalize_sampling_rate(waveform, samplerate)

        if segments:
            yield waveform, segments


annotated_dataset = Dataset.from_generator(
    read_wavfile_and_annotation,
    (tf.float32, tf.bool),
    (tf.TensorShape([None]), tf.TensorShape([None, N_FEATURES])))

segmented_dataset = Dataset.from_generator(
    read_wavfile_and_textgrid,
    (tf.float32, tf.bool, tf.string),
    (tf.TensorShape([None]),
     tf.TensorShape([None, N_FEATURES]),
     tf.TensorShape([])))

# next_element = dataset.make_one_shot_iterator().get_next()

# sess = tf.InteractiveSession()
# for i in range(10):
#     value = sess.run(next_element)
audio_dataset = Dataset.from_generator(
    read_wavfile,
    tf.float32,
    tf.TensorShape([None]))
