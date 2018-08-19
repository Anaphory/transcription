from spectrogram_to_sound import stft
import numpy
import tensorflow as tf
import matplotlib.pyplot as plt
from phonetic_features import features

from dataset import segmented_dataset
dataset = segmented_dataset.padded_batch(1, padded_shapes=([None], [None, 96], []))

iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                        dataset.output_shapes)
dataset_init_op = iterator.make_initializer(dataset)


waveforms, featurevectors, form = iterator.get_next()

magnitude_spectrograms = tf.abs(stft(waveforms))

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    sess.run(dataset_init_op)
    #    while True:
    for _ in range(5):
        try:
            f, feats, spects = sess.run((form,
                                         featurevectors,
                                         magnitude_spectrograms))
            print(f[0].decode("utf-8"))
            for feat, spec in zip(feats, spects):
                if not feat[:, features["vowel"]].any():
                    continue
                plt.subplot(2,1,1)
                plt.imshow(spec.T, cmap='hot', aspect='auto')
                plt.xlim(0, min(len(feat), len(spec.T))
                plt.subplot(2,1,2)
                for feature, pos in features.items():
                    if not feat[:, pos].any():
                        continue
                    if not feature in {"diphthong", "consonant", "vowel", "nasal", "fricative"}:
                        continue
                    plt.plot(range(len(feat)),
                             feat[:, pos] * numpy.random.random(),
                             label=feature)
                plt.xlim(0, min(len(feat), len(spec.T))
                plt.legend()
                print(feat.shape, spec.shape)
                plt.show()
        except tf.errors.OutOfRangeError:
            break

