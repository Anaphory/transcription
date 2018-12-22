import numpy

import keras
from keras.models import Model
from keras.layers import Dense, Input, Embedding
from keras.optimizers import Adadelta

from keras.utils import np_utils
from keras import backend as K

import dataset

# TEMP
import sys
sys.argv = ["emacs"]
# END TEMP

string_data = dataset.ChoppedStringSequence(chunk_size=1000, batch_size=1)

tokens = []
types = set()
for file, chunk in string_data.chunks:
    text = dataset.tokenizer(file.open().read()).split()
    for token in text:
        tokens.append(([
            numpy.array([x], dtype=int)
            for x in dataset.lookup_ipa(token)],
                       token))
        types.add(token)
types = list(types)

tokens = [(i, [np_utils.to_categorical(types.index(j), num_classes=len(types))])
          for i, j in tokens]

# Build the model
features = [Input(#name='feature'
    shape=[1], dtype='int64')
            for _ in dataset.features]

embed = [Embedding(len(s), 26)(f)
           for f, s in zip(features, dataset.features)]

connector = keras.layers.add(embed)
segment = Dense(units=len(types), activation='softmax')(connector)

ipa_from_features = Model(
    inputs=features,
    outputs=[segment])

ipa_from_features.compile(
    loss="categorical_crossentropy",
    optimizer=Adadelta())

for i in range(10):
    tokens.sort(key=lambda x: numpy.random.random())
    ipa_from_features.fit_generator(iter(tokens), steps_per_epoch=len(tokens))


def check(ipa_symbol):
    print(ipa_symbol)
    features = dataset.lookup_ipa(ipa_symbol)
    print([list(f)[v] for f, v in zip(dataset.features, features)])

    type = [(t, p)
            for t, p in zip(
                    types,
                    ipa_from_features.predict(
                        [numpy.array([v], int) for v in features])[0, 0])]
    type.sort(key=lambda x: x[1], reverse=True)
    print(type[:3])

for type in types:
    check(type)
