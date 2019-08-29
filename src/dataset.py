#!/usr/bin/env

"""Various sound datasets"""

# General packages
from pathlib import Path
import numpy
import bisect
import itertools

# Machine learning packages
import tensorflow as tf
from keras.utils import Sequence
from keras.preprocessing.sequence import pad_sequences

# Phonetic transcription packages
import pyclts
from feature_sound_class import FeatureSoundClasses
from segments import Tokenizer, Profile
import read_textgrid

# ‘Relative’ imports
from hparams import hparams

bipa = pyclts.TranscriptionSystem('bipa')
sounds = list(bipa.sounds)
sounds.extend([])
tokenizer = Tokenizer(Profile(*({"Grapheme": x, "mapping": x} for x in sounds)))


try:
    this = Path(__file__).absolute()
except NameError:
    this = Path("__file__").absolute()
DATA_PATH = this.parent.parent / "data" / "selection"

from lingpy.sequence.sound_classes import sampa2uni
from pyclts import SoundClasses, TranscriptionSystem

systems = [SoundClasses("asjp"),
           SoundClasses("sca"),
           SoundClasses("color"),
           SoundClasses("dolgo"),
           FeatureSoundClasses("height"),
           FeatureSoundClasses("duration"),
           SoundClasses("cv")]
features = [{c: i for i, c in enumerate([""] + sorted(system.classes))}
            for system in systems]

def lookup_ipa(ipasymbol):
    try:
        return [feature[system[bipa[ipasymbol]] or ""]
                for feature, system in zip(features, systems)]
    except (AssertionError, KeyError):
        return [0 for feature in features]

def lookup_sampa(sampasymbol):
    try:
        return lookup_ipa(sampa2uni(sampasymbol))
    except ValueError:
        return None

class TranscribedSequence(Sequence):
    def __init__(self, batch_size=10, files=None):
        if files is None:
            files = DATA_PATH.rglob("*.txt")
        self.batch_size = batch_size
        files.sort(key=lambda x: numpy.random.random())
        file_sizes = {}
        file_complexities = {}
        for f in range(len(files) - 1, -1, -1):
            file = files[f]
            try:
                sg = numpy.load(file.with_suffix(".npy").open("rb"))
            except (OSError, FileNotFoundError):
                from audio_prep import read_audio
                for sfx in ["wav", "ogg"]:
                    if file.with_suffix("."+sfx).exists():
                        sg = read_audio(file.with_suffix("."+sfx), sfx)
                        break
                else:
                    del files[f]
                    continue
                numpy.save(file.with_suffix(".npy").open("wb"), sg)
            file_sizes[file] = len(sg)
            file_complexities[file] = len(file.open().read().strip())
        avg_cx = numpy.median([s/cx for s, cx in zip(file_sizes.values(), file_complexities.values())])
        scale = max(file_sizes.values()) + avg_cx * max(file_complexities.values())
        files.sort(key=lambda x: file_sizes[x] + avg_cx*file_complexities[x] + numpy.random.random() * scale)
        print(files)
        self._len = 1
        self.files = files

    def __len__(self):
        return int(self._len)

    def on_epoch_end(self):
        with Path("check").open("a") as cf:
            cf.write("Ep {:04.2f} START\n".format(self._len))
        if self._len * self.batch_size > len(self.files):
            return
        self._len += 0.1

    @staticmethod
    def features_from_text(file, spectrogram_length):
        with file.open() as tr:
            text = tr.read()

        feature_strings = [
            []
            for feature in features]
        for segment in tokenizer(text, ipa=True).split():
            for feature_string, value in zip(
                    feature_strings, lookup_ipa(segment)):
                feature_string.append(value)
        return feature_strings

    def __getitem__(self, index):
        with Path("check").open("a") as cf:
            cf.write("Ep {:04.2f} Ix {:4d}\n".format(self._len, index))
        sgs = [
            numpy.load(file.with_suffix(".npy").open("rb"))
            for file in self.files[
                    index * self.batch_size: (index + 1) * self.batch_size]]
        spectrogram_lengths = numpy.zeros(len(sgs), dtype=int)
        spectrograms = numpy.zeros(
                (len(sgs),
                    int(max(len(s) for s in sgs) * (numpy.random.random() + 1)),
                    len(sgs[-1][-1])))
        for i, sg in enumerate(sgs):
            spectrograms[i][:len(sg)] = sg
            spectrogram_lengths[i] = len(sg)

        files = self.files[index * self.batch_size:
                            (index + 1) * self.batch_size]
        feature_values = [numpy.zeros(
            (len(spectrograms),
             237),
            dtype=int)
                            for f in features]
        feature_lengths = numpy.zeros(len(sgs), dtype=int)

        for i, file in enumerate(files):
            for feature_matrix, values in zip(
                    feature_values,
                    self.features_from_text(
                        file, 237)):
                feature_lengths[i] = feature_lengths[i] or len(values)
                feature_matrix[i][:feature_lengths[i]] = values

        return ([spectrograms] +
                 feature_values + [
                 spectrogram_lengths,
                 feature_lengths],
                [numpy.zeros(len(spectrograms)) for _ in features])

    def human_readable(self, index):
        return [
            (file.with_suffix(".txt"), file.with_suffix(".txt").open("r").read().strip())
            for file in self.files[
                    index * self.batch_size: (index + 1) * self.batch_size]]
