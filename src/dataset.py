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

import pyclts
import read_textgrid
import soundfile as sf
from normalize_sampling_rate import normalize_sampling_rate_windowed as normalize_sampling_rate

odd_symbols = {
    "H#": "#",
    "#H": "#",
    "I": "ɪ",
    "U": "ʊ",
    "6": "ɐ",
    "I6": "ɪɐ",
    "aI": "aɪ",
    "e:6": "e:ɐ",
    "@": "ə",
    "O6": "ɔɐ",
    "E6": "ɛɐ",
    "Z": "ʒ",
    "H": "h", # Actually, this is ʰ (aspiration), which however is not
              # an IPA segment.
    "A": "ɑ", # Maybe?
    "T": "θ",
    "E": "ɛ",
    "S": "ʃ",
    "C": "ç",
    "N": "ŋ",
    "Q": "ɒ",
    "tH": "tʰ",
    "V": "ʌ",
    "D": "ð",
    "=l": "l̩",
    "i@": "iə",
    "@:": "ə:",
    "OY": "ɔʏ",
    "Y": "ʏ",
    "*": "#",
    "o:C": "o:",
    }

# What do we know about phonetics?
ts = pyclts.TranscriptionSystem()
features = {"diphthong"}
for key in ts.features.keys():
    try:
        features |= key
    except TypeError:
        features.add(key)
features = {feature: f for f, feature in enumerate(features)}

# https://github.com/cldf/clts/blob/master/data/features.tsv lists 134
# feature values, plus we want one for 'Unknown'
N_FEATURES = 134 + 1

try:
    this = Path(__file__)
except NameError:
    this = Path("dataset.py").absolute()
DATA_PATH = this.parent.parent / "data"


def feature_vector_of_sound(sound):
    vector = numpy.zeros(N_FEATURES)
    if type(ts.resolve_sound(sound)) == pyclts.models.UnknownSound:
        return vector
    for feature in ts.resolve_sound(sound).featureset:
        vector[features[feature]] = 1
    return vector


def clean(sound):
    return odd_symbols.get(sound, sound)


def read_wavfile():
    for file in itertools.chain(DATA_PATH.glob("**/*.ogg"),
                                DATA_PATH.glob("**/*.wav")):
        waveform, samplerate = sf.read(file.open("rb"))
        if len(waveform.shape) > 1:
            waveform = waveform[:, 1]
        waveform = normalize_sampling_rate(waveform, samplerate)

        yield waveform


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


segmented_dataset = Dataset.from_generator(
    read_wavfile_and_annotation,
    (tf.float32, tf.bool),
    (tf.TensorShape([None]), tf.TensorShape([None, N_FEATURES])))

audio_dataset = Dataset.from_generator(
    read_wavfile,
    tf.float32,
    tf.TensorShape([None]))
