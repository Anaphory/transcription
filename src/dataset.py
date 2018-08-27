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
        yield file


def read_wavefile_normalized_mono(file):
    waveform, samplerate = librosa.load(file, sr=hparams.sample_rate)
    if len(waveform.shape) > 1:
        waveform = waveform[:, 1]

    return waveform


def wavfile_with_textgrid():
    for file in list_wavfiles():
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

        for start, end, segment in phonetics.simple_transcript:
            start = float(start)
            end = float(end)
            if transform:
                segment = transform(segment)
            form += segment
            for window in range(int(start * windows_per_second),
                                int(end * windows_per_second)):
                feature_matrix[window] = window_features

        yield file, segments


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

        waveform = read_wavefile_normalized_mono(file)

        if segments:
            yield waveform, segments


audio_dataset = Dataset.from_generator(
    list_wavfiles,
    tf.string,
    tf.TensorShape([1]))
