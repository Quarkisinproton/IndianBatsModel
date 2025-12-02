import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import librosa
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_wombat_json(path: Path) -> Dict:
    with path.open('r') as f:
        return json.load(f)


def find_audio_for_json(json_path: Path, raw_audio_dirs: List[Path]) -> Optional[Path]:
    data = load_wombat_json(json_path)
    # common field containing recording path
    rec = None
    if isinstance(data, dict):
        rec = data.get('recording') or data.get('audio_file') or data.get('file')
    
    if rec:
        p = Path(rec)
        if p.is_absolute() and p.exists():
            return p
        # try relative to any raw_audio_dir
        for d in raw_audio_dirs:
            candidate = d / p.name
            if candidate.exists():
                return candidate

    # fallback: match by stem name
    stem = json_path.stem
    for d in raw_audio_dirs:
        for ext in ('.wav', '.flac', '.mp3', '.m4a'):
            cand = d / (stem + ext)
            if cand.exists():
                return cand
        # try scanning raw_audio_dir for a file containing the stem
        for f in d.glob('*'):
            if stem in f.stem:
                return f
    return None


def extract_segment(y: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    start = max(0, int(start_s * sr))
    end = min(len(y), int(end_s * sr))
    return y[start:end]


def make_mel_spectrogram(y: np.ndarray, sr: int, n_mels: int = 128, hop_length: int = 512) -> np.ndarray:
    if y.size == 0:
        return np.zeros((n_mels, 1), dtype=float)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def save_spectrogram_image(S_db: np.ndarray, out_path: Path, cmap: str = 'magma', dpi: int = 100) -> None:
    # ensure parent exists
    ensure_dir(out_path.parent)
    plt.figure(figsize=(3, 3))
    plt.axis('off')
    plt.imshow(S_db, aspect='auto', origin='lower', cmap=cmap)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()


def normalize_annotations(raw_anns) -> List[Dict]:
    # wombat exports vary; normalize to list of dicts with start, end, label
    if raw_anns is None:
        return []
    if isinstance(raw_anns, dict):
        # maybe a dict of segments
        return [raw_anns]
    if isinstance(raw_anns, list):
        return raw_anns
    return []


def process_audio_file(audio_path: Path, annotations: Iterable[Dict], out_base: Path, species_key: str = 'label') -> None:
    y, sr = librosa.load(str(audio_path), sr=None)
    for i, ann in enumerate(annotations):
        # common field names
        start = ann.get('start_time') or ann.get('start') or ann.get('t0') or ann.get('onset')
        end = ann.get('end_time') or ann.get('end') or ann.get('t1') or ann.get('offset')
        label = ann.get(species_key) or ann.get('species') or ann.get('label') or ann.get('class')
        if start is None or end is None or label is None:
            # try flexible keys
            # skip if insufficient data
            continue
        try:
            start_f = float(start)
            end_f = float(end)
        except Exception:
            continue
        seg = extract_segment(y, sr, start_f, end_f)
        if seg.size == 0:
            continue
        S_db = make_mel_spectrogram(seg, sr)
        safe_label = str(label).strip().replace(' ', '_')
        out_dir = out_base / safe_label
        ensure_dir(out_dir)
        out_name = f"{audio_path.stem}_{i}.png"
        out_path = out_dir / out_name
        save_spectrogram_image(S_db, out_path)


def process_all(raw_audio_dirs: List[str], json_dir: str, out_dir: str, species_key: str = 'label') -> None:
    raw_audio_dirs = [Path(d) for d in raw_audio_dirs]
    json_dir = Path(json_dir)
    out_base = Path(out_dir)
    ensure_dir(out_base)

    for jpath in json_dir.rglob('*.json'):
        try:
            data = load_wombat_json(jpath)
        except Exception:
            continue
        audio_path = find_audio_for_json(jpath, raw_audio_dirs)
        if audio_path is None:
            # skip if we can't find the audio
            continue

        # annotations might be top-level or under keys
        anns = None
        if isinstance(data, dict):
            for key in ('annotations', 'labels', 'segments', 'events'):
                if key in data:
                    anns = data[key]
                    break
            if anns is None:
                # maybe the JSON itself is a list-like mapping
                if any(k in data for k in ('start_time', 'end_time', species_key)):
                    anns = [data]
        else:
            anns = data

        anns = normalize_annotations(anns)
        process_audio_file(audio_path, anns, out_base, species_key=species_key)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert Wombat JSON annotations + audio -> spectrogram PNGs')
    parser.add_argument('--raw_audio_dir', required=True, nargs='+', help='directory with raw audio files')
    parser.add_argument('--json_dir', required=True, help='directory with Wombat JSON exports')
    parser.add_argument('--out_dir', required=True, help='output directory for spectrograms')
    parser.add_argument('--species_key', default='label', help='JSON key for species label')
    args = parser.parse_args()
    process_all(args.raw_audio_dir, args.json_dir, args.out_dir, species_key=args.species_key)
