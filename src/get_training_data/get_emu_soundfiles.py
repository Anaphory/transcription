import soundfile as sf
from pathlib import Path
import shutil

emupath = Path("/home/gereon/Downloads/emu")


def parse_lab_file_to_praat_textgrid(lab_file, audio_length):
    blobs = ""
    old_start = 0
    old_text = ""
    for l, line in enumerate(lab_file.open("r")):
        line = line.strip()
        if l == 0:
            assert line.startswith("signal")
        elif l == 1:
            assert line == "nfields 1"
        elif l == 2:
            assert line == "#"
        else:
            start, _, phone = line.split()
            start = float(start)
            blobs += '''
      intervals [{no:}]:
          xmin = {start:}
          xmax = {end:}
          text = "{text:}"'''.format(no=l - 1, start=old_start, end=start, text=phone)
            old_start = start

    return "\n".join(["""File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = {xmax:}
tiers? <exists>
size = 1
item []:
    item [1]:
       class = "IntervalTier"
       name = "Phonetic"
       xmin = 0
       xmax = {xmax:}
       intervals: size = {len_intervals:}""".format(
           len_intervals=l - 2,
           xmax=audio_length),
                     blobs,
                     ('''      intervals [{no:}]:
          xmin = {start:}
          xmax = {end:}
          text = "{text:}"'''.format(
              no=l - 2, start=old_start, end=audio_length, text="#H"))])
                     

target = Path("/home/gereon/devel/transcription/data/emu/")
target.mkdir(exist_ok=True)
for corpus in (emupath / "corpora").glob("*"):
    if corpus.is_dir():
        for wav_file in corpus.glob("**/*.wav"):
            files = set()
            d, fr = sf.read(wav_file.open("rb"))
            for possible_label_file in corpus.glob(
                    "**/{:}.*".format(wav_file.stem)):
                if possible_label_file.suffix in (".lab", ".ph"):
                    (target / (wav_file.stem + ".txt")).open("w").write(
                        parse_lab_file_to_praat_textgrid(
                            possible_label_file,
                            len(d) / fr))
                    shutil.copy(wav_file, target)
                    break
                try:
                    files.add(possible_label_file)
                    content = possible_label_file.open().read()
                except UnicodeDecodeError:
                    continue
            else:
                print(wav_file)
                print(files)
                    
