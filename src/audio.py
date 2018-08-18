import os
import sys
import tempfile
import subprocess

try:
    import pyaudio

    # instantiate PyAudio (1)

    def pyaudio_play(input_array):
        """Play a numpy array as audio"""
        p = pyaudio.PyAudio()

        # open stream (2), 2 is size in bytes of int16
        stream = p.open(format=p.get_format_from_width(2),
                        channels=1,
                        rate=44100,
                        output=True)

        # play stream (3), blocking call
        stream.write(input_array)

        # stop stream (4)
        stream.stop_stream()
        stream.close()

        p.terminate()
except ImportError:
    pass


def x_play_file(file, player="mplayer"):
    player = subprocess.Popen([player, file],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    print(player.stderr.read().decode('utf-8'),
          file=sys.stdout)


try:
    import numpy
    import scipy.io.wavfile

    def scipy_play(input_sequence, player="mplayer", rate=44100):
        """Play the audio signal described by input_sequence.

        The input sequence is assumed to describe an audio signal. Its first
        dimension corresponds to the audio samples (single numbers or sequences of
        1 or 2 numbers), and is assumed to have values 0 <= v < 256 which are
        truncated and modulo-ed to numpy.uint8 before writing to a sound file and
        played with an external player.

        Parameters
        ----------
        input_sequence: sequence
            The audio samples.

        rate: int
            The sampling rate of the audio.

        """
        input_array = numpy.array(input_sequence, dtype=numpy.uint8)
        print(input_array)
        _, path = tempfile.mkstemp(suffix=".wav")
        with open(path, "wb") as wavfile:
            scipy.io.wavfile.write(wavfile, rate, input_array)

        x_play_file(path, player)
        os.remove(path)
except ImportError:
    pass
