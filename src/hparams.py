#!/usr/bin/python

"""Model hyperparameters."""

import argparse
parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--n-lstm-hidden", default=[50],
                    type=lambda x: [int(i) for i in x.split()],
                    help="Hidden layer sizes as whitespace-separated"
                    "list of integers (default: 100 100 150)")
parser.add_argument("--n-spectrogram", default=1025,
                    type=int,
                    help="Size of the spectrogram, ideally 2^N+1."
                    "(default: 513)")
parser.add_argument("--frame-length-ms", default=5,
                    type=int,
                    help="Length of each window in ms"
                    "(default: 5)")
parser.add_argument("--frame-shift-ms", default=3,
                    type=int,
                    help="Distance between two subsequent windows in ms"
                    "(default: 3)")
args = parser.parse_args()

hparams = {
    "n_spectrogram": args.n_spectrogram,
    "n_lstm_hidden": args.n_lstm_hidden,
    "frame_length_ms": args.frame_length_ms,
    "frame_shift_ms": args.frame_shift_ms,
    "n_features_hidden": 40,
    "prediction_time_at_least_ms": 40,
    "sample_rate": 44100,
    "griffin_lim_iters": 60,
    "max_string_length": 160,
    "chop": 2,
}
