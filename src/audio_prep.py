
g = tf.Graph()
with g.as_default():
    waveform = tf.squeeze(
        tf.contrib.ffmpeg.decode_audio(
            tf.read_file(tf.placeholder(tf.string, name="filename")),
            file_format=tf.placeholder(tf.string, name="format"),
            samples_per_second=hparams["sample_rate"],
            channel_count=1))

    log_mag_spectrogram = tf.log(tf.abs(stft(waveform) + 1e-8))



def read_audio(filename, format):
    with tf.Session(graph=g) as sess:
        return sess.run(log_mag_spectrogram, feed_dict={
            'filename:0': str(filename),
            'format:0': format})

