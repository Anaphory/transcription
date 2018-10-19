#!/usr/bin/python

"""Model hyperparameters."""

hparams = {
    "n_spectrogram": 513,
    "n_lstm_hidden": [100, 100, 150],
    "n_features_hidden": 40,
    "frame_length_ms": 15,
    "frame_shift_ms": 5,
    "prediction_time_at_least_ms": 40,
    "sample_rate": 44100,
    "griffin_lim_iters": 60,
}
