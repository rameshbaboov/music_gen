import math
import configparser
import sys
import os
import wave
import struct

import numpy as np
import sounddevice as sd

# These will be loaded from constants file
SAMPLE_RATE = None
C4_FREQ = None
SARGAM_TO_RATIO = {}
NOTE_TO_SEMITONE = {}
BASE_CHORDS = {}
DEFAULT_OCTAVE = 4  # fallback if not set in constants


# ---------- CONSTANTS LOADING ----------

def pick_section(cfg, base: str, profile: str) -> str:
    """
    Try <base>:<profile>, else fall back to <base>.
    Raises if neither exists.
    """
    prof_section = f"{base}:{profile}"
    if prof_section in cfg:
        return prof_section
    if base in cfg:
        return base
    raise ValueError(f"No section [{prof_section}] or [{base}] found in constants file")

def clean_value(raw: str) -> str:
    """Strip inline comments and whitespace."""
    return raw.split(";", 1)[0].split("#", 1)[0].strip()


def parse_number(raw: str) -> float:
    """Parse float or simple fraction like 9/8."""
    raw = clean_value(raw)
    if "/" in raw:
        num, den = raw.split("/", 1)
        return float(num.strip()) / float(den.strip())
    return float(raw)

def load_constants(constants_path: str,
                   audio_profile: str,
                   tuning_profile: str,
                   sargam_profile: str,
                   note_profile: str,
                   chord_profile: str):
    global SAMPLE_RATE, C4_FREQ, SARGAM_TO_RATIO, NOTE_TO_SEMITONE, BASE_CHORDS, DEFAULT_OCTAVE

    if not os.path.exists(constants_path):
        raise FileNotFoundError(f"Constants file not found: {constants_path}")

    cfg = configparser.ConfigParser()
    cfg.optionxform = str  # preserve case
    cfg.read(constants_path)

    # ---- AUDIO ----
    audio_sec = pick_section(cfg, "audio", audio_profile)
    if "sample_rate" not in cfg[audio_sec]:
        raise ValueError(f"[{audio_sec}]/sample_rate missing in constants file")
    SAMPLE_RATE = int(clean_value(cfg[audio_sec]["sample_rate"]))

    # ---- TUNING ----
    tuning_sec = pick_section(cfg, "tuning", tuning_profile)
    if "c4_freq" not in cfg[tuning_sec]:
        raise ValueError(f"[{tuning_sec}]/c4_freq missing in constants file")
    C4_FREQ = parse_number(cfg[tuning_sec]["c4_freq"])

    if cfg.has_option(tuning_sec, "default_octave"):
        DEFAULT_OCTAVE = int(clean_value(cfg[tuning_sec]["default_octave"]))
    else:
        DEFAULT_OCTAVE = 4

    # ---- SARGAM ----
    sargam_sec = pick_section(cfg, "sargam", sargam_profile)
    SARGAM_TO_RATIO = {}
    for name, val in cfg[sargam_sec].items():
        SARGAM_TO_RATIO[name.strip().lower()] = parse_number(val)

    # ---- NOTE → SEMITONE ----
    note_sec = pick_section(cfg, "note_semitone", note_profile)
    NOTE_TO_SEMITONE = {}
    for name, val in cfg[note_sec].items():
        NOTE_TO_SEMITONE[name.strip()] = int(clean_value(val))

    # ---- CHORDS ----
    chord_sec = pick_section(cfg, "chords", chord_profile)
    BASE_CHORDS = {}
    for chord_name, notes_str in cfg[chord_sec].items():
        chord_name = chord_name.strip()
        notes = [n.strip() for n in clean_value(notes_str).split(",") if n.strip()]
        BASE_CHORDS[chord_name] = notes

    # assign globals
    globals()["SAMPLE_RATE"] = SAMPLE_RATE
    globals()["C4_FREQ"] = C4_FREQ
    globals()["SARGAM_TO_RATIO"] = SARGAM_TO_RATIO
    globals()["NOTE_TO_SEMITONE"] = NOTE_TO_SEMITONE
    globals()["BASE_CHORDS"] = BASE_CHORDS
    globals()["DEFAULT_OCTAVE"] = DEFAULT_OCTAVE

# ---------- AUDIO HELPERS ----------

def midi_to_freq(midi_note: int) -> float:
    return 440.0 * (2 ** ((midi_note - 69) / 12))


def note_name_to_freq(note: str) -> float:
    """Accepts 'C4' or just 'C' (then uses DEFAULT_OCTAVE)."""
    note = note.strip()
    pitch = "".join(c for c in note if not c.isdigit())
    octave_str = "".join(c for c in note if c.isdigit())

    if not pitch or pitch not in NOTE_TO_SEMITONE:
        raise ValueError(f"Unsupported note pitch: {note}")

    if octave_str:
        octave = int(octave_str)
    else:
        octave = DEFAULT_OCTAVE  # default if no octave given

    semitone = NOTE_TO_SEMITONE[pitch]
    midi = (octave + 1) * 12 + semitone
    return midi_to_freq(midi)


def generate_wave(freq: float, duration: float, waveform: str, volume: float) -> list:
    n_samples = int(SAMPLE_RATE * duration)
    data = []
    for i in range(n_samples):
        t = i / SAMPLE_RATE
        if waveform == "sine":
            x = math.sin(2 * math.pi * freq * t)
        elif waveform == "cos":
            x = math.cos(2 * math.pi * freq * t)
        elif waveform == "square":
            x = 1.0 if math.sin(2 * math.pi * freq * t) >= 0 else -1.0
        elif waveform == "saw":
            x = 2 * (t * freq - math.floor(0.5 + t * freq))
        else:
            x = math.sin(2 * math.pi * freq * t)
        data.append(volume * x)
    return data


def mix_waves(waves, volume: float):
    n_samples = max(len(w) for w in waves)
    mixed = [0.0] * n_samples
    for w in waves:
        for i, s in enumerate(w):
            mixed[i] += s
    max_amp = max(abs(s) for s in mixed) or 1.0
    return [s / max_amp * volume for s in mixed]


def sargam_to_freq(sargam: str) -> float:
    s = sargam.strip().lower()
    if s not in SARGAM_TO_RATIO:
        raise ValueError(f"Unsupported sargam note: {sargam}")
    return C4_FREQ * SARGAM_TO_RATIO[s]


def apply_transpose(freq: float, transpose_semitones: int) -> float:
    if transpose_semitones == 0:
        return freq
    return freq * (2 ** (transpose_semitones / 12.0))


def token_to_wave(
    token: str,
    chords: dict,
    duration: float,
    waveform: str,
    volume: float,
    transpose_semitones: int,
):
    token = token.strip()
    low = token.lower()

    # sargam
    if low in SARGAM_TO_RATIO:
        f = sargam_to_freq(low)
        f = apply_transpose(f, transpose_semitones)
        return generate_wave(f, duration, waveform, volume)

    # chord
    if token in chords:
        freqs = [note_name_to_freq(n) for n in chords[token]]
        freqs = [apply_transpose(f, transpose_semitones) for f in freqs]
        waves = [generate_wave(f, duration, waveform, volume) for f in freqs]
        return mix_waves(waves, volume)

    # western single note
    f = note_name_to_freq(token)
    f = apply_transpose(f, transpose_semitones)
    return generate_wave(f, duration, waveform, volume)


def sequence_to_wave(
    tokens,
    chords: dict,
    bpm: float,
    note_length: float,
    waveform: str,
    volume: float,
    transpose_semitones: int,
):
    seconds_per_beat = 60.0 / bpm
    duration = seconds_per_beat * note_length

    full = []
    for t in tokens:
        if not t.strip():
            continue
        w = token_to_wave(
            t, chords, duration, waveform, volume, transpose_semitones
        )
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


def play_audio(samples):
    data = np.array(samples, dtype="float32")
    sd.play(data, SAMPLE_RATE)
    sd.wait()


# ---------- SONG INI LOADING ----------

def load_song_ini(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Song INI not found: {path}")

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(path)

    if "sequence" not in config or "tokens" not in config["sequence"]:
        raise ValueError(f"No [sequence]/tokens in {path}")

    raw = config["sequence"]["tokens"].strip()
    raw = raw.replace("\n", " ")
    tokens = [t.strip() for t in raw.split(",") if t.strip()]

    chords = dict(BASE_CHORDS)
    if "chords" in config:
        for name, value in config.items("chords"):
            chord_name = name.strip()
            notes_str = clean_value(value)
            notes = [n.strip() for n in notes_str.split(",") if n.strip()]
            if notes:
                chords[chord_name] = notes

    return tokens, chords


# ---------- CLEAN NUMERIC READERS FOR MAIN CONFIG ----------

def get_float_clean(cfg, section, option, fallback):
    if not cfg.has_option(section, option):
        return fallback
    raw = clean_value(cfg.get(section, option))
    if not raw:
        return fallback
    try:
        return float(raw)
    except ValueError:
        return fallback


def get_int_clean(cfg, section, option, fallback):
    if not cfg.has_option(section, option):
        return fallback
    raw = clean_value(cfg.get(section, option))
    if not raw:
        return fallback
    try:
        return int(raw)
    except ValueError:
        return fallback


# ---------- CONFIG PROCESSING ----------

def process_song_ini(
    song_ini_path: str,
    waveform: str,
    bpm: float,
    note_length: float,
    volume: float,
    transpose_semitones: int,
    output_mode: str,
):
    tokens, chords = load_song_ini(song_ini_path)
    samples = sequence_to_wave(
        tokens, chords, bpm, note_length, waveform, volume, transpose_semitones
    )
    base = os.path.splitext(os.path.basename(song_ini_path))[0]
    out_name = base + ".wav"

    if output_mode in ("wav", "both"):
        save_wav(samples, out_name)
        print(f"Generated {out_name} from {song_ini_path}")

    if output_mode in ("play", "both"):
        print(f"Playing {base} ...")
        play_audio(samples)


def process_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(config_path)

    # constants file
    if "constants" not in cfg or "constants_file" not in cfg["constants"]:
        raise ValueError("config.ini must have [constants]/constants_file")
    base_dir = os.path.dirname(os.path.abspath(config_path))
   
    constants_file = clean_value(cfg["constants"]["constants_file"])
    constants_path = os.path.join(base_dir, constants_file)

    audio_profile  = clean_value(cfg["constants"].get("audio_profile",  "default"))
    tuning_profile = clean_value(cfg["constants"].get("tuning_profile", "standard"))
    sargam_profile = clean_value(cfg["constants"].get("sargam_profile", "carnatic"))
    note_profile   = clean_value(cfg["constants"].get("note_profile",   "western"))
    chord_profile  = clean_value(cfg["constants"].get("chord_profile",  "basic"))

    load_constants(constants_path,
                audio_profile,
                tuning_profile,
                sargam_profile,
                note_profile,
                chord_profile)

    # output mode: play | wav | both
    if "output" in cfg and "mode" in cfg["output"]:
        output_mode = clean_value(cfg["output"]["mode"]).lower()
    else:
        output_mode = "play"
    if output_mode not in ("play", "wav", "both"):
        output_mode = "play"

    if "songs" not in cfg:
        raise ValueError("config.ini must have a [songs] section")

    for song_ini, flag in cfg["songs"].items():
        if flag.strip().lower() not in ("yes", "true", "1", "y"):
            continue

        song_ini = song_ini.strip()
        song_path = os.path.join(base_dir, song_ini)

        # per-song section name = ini file name
        section = song_ini
        waveform = "sine"
        bpm = 120.0
        note_length = 1.0
        volume = 0.5
        transpose_semitones = 0

        if section in cfg:
            val = cfg.get(section, "waveform", fallback=waveform)
            waveform = clean_value(val) or waveform
            bpm = get_float_clean(cfg, section, "bpm", bpm)
            note_length = get_float_clean(cfg, section, "note_length", note_length)
            volume = get_float_clean(cfg, section, "volume", volume)
            transpose_semitones = get_int_clean(
                cfg, section, "transpose_semitones", transpose_semitones
            )

        process_song_ini(
            song_path, waveform, bpm, note_length, volume, transpose_semitones, output_mode
        )


# ---------- MAIN ----------

if __name__ == "__main__":
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "config.ini"
    process_config(cfg_file)
