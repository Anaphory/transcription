from pyclts.transcriptionsystem import Sound, TranscriptionSystem
from pyclts.soundclasses import SoundClasses
from pyclts.util import read_data

class FeatureSoundClasses(SoundClasses):
    """A sond class for phonetic feature values"""
    def __init__(self, feature):
        """Create a sound class for 'feature'"""
        data, self.sounds, self.names = read_data(
            'soundclasses', 'lingpy.tsv', 'CLTS_NAME')
        self.system = TranscriptionSystem('bipa')
        self.feature = feature
        self.data = {}
        self.classes = set()
        for key, sound in data.items():
            try:
                value = getattr(self.system[key], feature)
            except AttributeError:
                value = "N/A"
            if value is None:
                value = "0"
            self.data[key] = {"grapheme": value}
            self.classes.add(value)

    def resolve_sound(self, sound):
        """Function tries to identify a sound in the data.

        Notes
        -----
        The function tries to resolve sounds to take a sound with less complex
        features in order to yield the next approximate sound class, if the
        transcription data are sound classes.
        """
        sound = sound if isinstance(sound, Sound) else self.system[sound]
        if sound.name in self.data:
            return self.data[sound.name]['grapheme']
        if not sound.type == 'unknownsound':
            if sound.type in ['diphthong', 'cluster']:
                return self.resolve_sound(sound.from_sound)
            try:
                return getattr(sound, self.feature)
            except AttributeError:
                return "N/A"
