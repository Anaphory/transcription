#!/usr/bin/env python

"""Download audio and transcription files from soundcomparisons.com."""

import re
import json
from pathlib import Path
import urllib.request

URL_TEMPLATE="https://soundcomparisons.com/query/data?study={study}"

try:
    PATH = Path(__file__)
except NameError:
    PATH = Path("./get_soundcomparison_files.py")
PATH = PATH.absolute().parent.parent.parent / "data" / "soundcomparisons"

def get_table_of_contents(study):
    with urllib.request.urlopen(
            URL_TEMPLATE.format(study=study)) as v:
        return json.load(v)

for study in ["Germanic"]:
    for id, form in get_table_of_contents(study)["transcriptions"].items():
        if "Phonetic" in form and "soundPaths" in form:
            if type(form["Phonetic"]) == str:
                form["Phonetic"] = [form["Phonetic"]]
                form["soundPaths"] = [form["soundPaths"]]
            for ipa, sound_paths in zip(form["Phonetic"],
                                        form["soundPaths"]):
                print(id)
                sound_paths.sort(
                    key=lambda file:
                    {"wav": 0, "ogg": 1, "mp3": 2}.get(file[-3:], 3))
                try:
                    url = sound_paths[0]
                except IndexError:
                    continue
                if not ipa:
                    continue
                name = Path(url)
                with urllib.request.urlopen(url) as remotesoundfile:
                    with open(PATH / name.name, "wb") as localsoundfile:
                        localsoundfile.write(remotesoundfile.read())
                with open(PATH / (name.stem + '.txt'), "w") as transcription:
                    transcription.write(ipa)
        else:
            print(id, form.keys())


