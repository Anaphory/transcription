import pyclts
import numpy


odd_symbols = {
    "H#": "#",
    "#H": "#",
    "H": "h", # Actually, this is ʰ (aspiration), which however is not
              # an IPA segment.
    "tH": "tʰ",
}

xsampa_substitutions = {
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
    "A": "ɑ",
    "T": "θ",
    "E": "ɛ",
    "S": "ʃ",
    "C": "ç",
    "N": "ŋ",
    "Q": "ɒ",
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
ts = pyclts.TranscriptionSystem(id_="bipa")
features = {"diphthong"}
for key in ts.features.keys():
    if 'marker' in key:
        continue
    try:
        features |= key
    except TypeError:
        features.add(key)
features = {feature: f for f, feature in enumerate(features)}

# https://github.com/cldf/clts/blob/master/data/features.tsv lists 134
# feature values, plus we want one for 'Unknown'
N_FEATURES = len(features)


def feature_vector_of_sound(sound):
    vector = numpy.zeros(N_FEATURES)
    if type(ts.resolve_sound(sound)) == pyclts.models.UnknownSound:
        return vector
    for feature in ts.resolve_sound(sound).featureset:
        try:
            vector[features[feature]] = 1
        except KeyError:
            continue
    return vector


def xsampa(string):
    string = odd_symbols.get(string, string)
    string = xsampa_substitutions.get(string, string)
    return string
