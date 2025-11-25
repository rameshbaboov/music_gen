import math
import wave
import struct
import configparser
import sys
import os

SAMPLE_RATE = 44100
DURATION_PER_NOTE = 0.5  # seconds
VOLUME = 0.5

# ---------- NOTE DEFINITIONS ----------

C4_FREQ = 261.63  # base for "sa"

SARGAM_TO_RATIO = {
    "sa": 1.0,
    "re": 9 / 8,
    "ga": 5 / 4,
    "ma": 4 / 3,
    "pa": 3 / 2,
    "da": 5 / 3,
    "ne": 15 / 8,
}

NOTE_TO_SEMITONE = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11,
}

# Base English chords (can be overridden per song INI)
BASE_CHORDS = {
    # Major
    "C":  ["C4", "E4", "G4"],
    "D":  ["D4", "F#4", "A4"],
    "E":  ["E4", "G#4", "B4"],
    "F":  ["F4", "A4", "C5"],
    "G":  ["G3", "B3", "D4"],
    "A":  ["A3", "C#4", "E4"],
    "B":  ["B3", "D#4", "F#4"],

    # Minor
    "Cm": ["C4", "D#4", "G4"],
    "Dm": ["D4", "F4", "A4"],
    "Em": ["E4", "G4", "B4"],
    "Fm": ["F4", "G#4", "C5"],
    "Gm": ["G3", "A#3", "D4"],
    "Am": ["A3", "C4", "E4"],
    "Bm": ["B3", "D4", "F#4"],
}


# ---------- AUDIO HELPERS ----------

def midi_to_freq(midi_note: int) -> float:
    return 440.0 * (2 ** ((midi_note - 69) / 12))


def note_name_to_freq(note: str) -> float:
    note = note.strip()
    pitch = "".join(c for c in note if not c.isdigit())
    octave_str = "".join(c for c in note if c.isdigit())
    if pitch not in NOTE_TO_SEMITONE or not octave_str:
        raise ValueError(f"Unsupported note format: {note}")
    octave = int(octave_str)
    semitone = NOTE_TO_SEMITONE[pitch]
    midi = (octave + 1) * 12 + semitone
    return midi_to_freq(midi)


def generate_sine(freq: float, duration: float) -> list:
    n_samples = int(SAMPLE_RATE * duration)
    return [
        VOLUME * math.sin(2 * math.pi * freq * (i / SAMPLE_RATE))
        for i in range(n_samples)
    ]


def mix_waves(waves):
    n_samples = max(len(w) for w in waves)
    mixed = [0.0] * n_samples
    for w in waves:
        for i, s in enumerate(w):
            mixed[i] += s
    max_amp = max(abs(s) for s in mixed) or 1.0
    return [s / max_amp * VOLUME for s in mixed]


def sargam_to_freq(sargam: str) -> float:
    s = sargam.strip().lower()
    if s not in SARGAM_TO_RATIO:
        raise ValueError(f"Unsupported sargam note: {sargam}")
    return C4_FREQ * SARGAM_TO_RATIO[s]


def token_to_wave(token: str, chords: dict):
    token = token.strip()
    low = token.lower()

    # sargam note
    if low in SARGAM_TO_RATIO:
        f = sargam_to_freq(low)
        return generate_sine(f, DURATION_PER_NOTE)

    # chord
    if token in chords:
        freqs = [note_name_to_freq(n) for n in chords[token]]
        waves = [generate_sine(f, DURATION_PER_NOTE) for f in freqs]
        return mix_waves(waves)

    # western single note
    f = note_name_to_freq(token)
    return generate_sine(f, DURATION_PER_NOTE)


def sequence_to_wave(tokens, chords: dict):
    full = []
    for t in tokens:
        if not t.strip():
            continue
        w = token_to_wave(t, chords)
        full.extend(w)
    return full


def save_wav(samples, filename: str):
    with wave.open(filename, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        for s in samples:
            val = int(max(-1.0, min(1.0, s)) * 32767)
            wf.writeframes(struct.pack("<h", val))


# ---------- INI LOADING ----------

def load_song_ini(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Song INI not found: {path}")

    config = configparser.ConfigParser()
    config.optionxform = str  # keep key case (Cm, Gm, etc.)
    config.read(path)

    if "sequence" not in config or "tokens" not in config["sequence"]:
        raise ValueError(f"No [sequence]/tokens in {path}")

    raw = config["sequence"]["tokens"].strip()
    # Allow multiline, comments via ';' handled by ConfigParser already
    raw = raw.replace("\n", " ")
    tokens = [t.strip() for t in raw.split(",") if t.strip()]

    # Build chords for this song: base + overrides
    chords = dict(BASE_CHORDS)
    if "chords" in config:
        for name, value in config.items("chords"):
            chord_name = name.strip()
            notes = [n.strip() for n in value.split(",") if n.strip()]
            if notes:
                chords[chord_name] = notes

    return tokens, chords


# ---------- CONFIG PROCESSING ----------

def process_song_ini(song_ini_path: str):
    tokens, chords = load_song_ini(song_ini_path)
    samples = sequence_to_wave(tokens, chords)
    base = os.path.splitext(os.path.basename(song_ini_path))[0]
    out_name = base + ".wav"
    save_wav(samples, out_name)
    print(f"Generated {out_name} from {song_ini_path}")


def process_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(config_path)

    if "songs" not in cfg:
        raise ValueError("config.ini must have a [songs] section")

    base_dir = os.path.dirname(os.path.abspath(config_path))

    for song_ini, flag in cfg["songs"].items():
        if flag.strip().lower() in ("yes", "true", "1", "y"):
            song_ini = song_ini.strip()
            song_path = os.path.join(base_dir, song_ini)
            process_song_ini(song_path)


# ---------- MAIN ----------

if __name__ == "__main__":
    # Usage: python script.py [config.ini]
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "config.ini"
    process_config(cfg_file)
