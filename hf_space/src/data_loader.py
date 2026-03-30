# ================================================================
# src/data_loader.py — Dataset loading for RAVDESS, TESS, CREMA-D
# ================================================================

import os, glob
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (RAVDESS_DIR, TESS_DIR, CREMAD_DIR,
                    RAVDESS_MAP, TESS_MAP, CREMAD_MAP, EMOTIONS)


def load_ravdess(data_dir=RAVDESS_DIR):
    """
    Load RAVDESS speech AND song files.
    Filename format: 03-01-05-01-01-01-01.wav
      field[0] = modality: 03=speech-only, 04=song
      field[2] = emotion code (mapped via RAVDESS_MAP)
      field[6] = actor ID (01-24)

    Accepts both modality 03 (speech) and 04 (song) — same emotion
    codes, same actors, doubles training data for non-test speakers.
    """
    records = []
    files   = glob.glob(os.path.join(data_dir, '**', '*.wav'), recursive=True)
    if not files:
        print(f"⚠️  No RAVDESS files in {data_dir}")
        print("   Run: python src/download_data.py")
        return pd.DataFrame()

    for fp in files:
        parts = os.path.basename(fp).replace('.wav', '').split('-')
        # Accept modality 03 (speech) and 04 (song)
        if (len(parts) == 7
                and parts[0] in ('03', '04')
                and parts[2] in RAVDESS_MAP):
            records.append({
                'filepath': fp,
                'emotion':  RAVDESS_MAP[parts[2]],
                'actor_id': int(parts[6]),
                'dataset':  'ravdess',
            })

    df = pd.DataFrame(records)
    if len(df) > 0:
        print(f"✅ RAVDESS: {len(df)} samples, {df['actor_id'].nunique()} actors")
    else:
        print("⚠️  RAVDESS: no usable samples found")
    return df


def load_tess(data_dir=TESS_DIR):
    records = []
    files   = glob.glob(os.path.join(data_dir, '**', '*.wav'), recursive=True)
    if not files:
        files = glob.glob(os.path.join(data_dir, '**', '*.WAV'), recursive=True)
    if not files:
        print(f"⚠️  No TESS files in {data_dir}")
        return pd.DataFrame()

    actor_map = {'OAF': 101, 'YAF': 102}
    for fp in files:
        parts    = os.path.basename(fp).lower().replace('.wav', '').split('_')
        emo      = TESS_MAP.get(parts[-1]) if parts else None
        speaker  = parts[0].upper() if parts else ''
        actor_id = actor_map.get(speaker, 100)
        if emo and emo in EMOTIONS:
            records.append({'filepath': fp, 'emotion': emo,
                            'actor_id': actor_id, 'dataset': 'tess'})

    df = pd.DataFrame(records)
    print(f"✅ TESS: {len(df)} samples")
    return df


def load_cremad(data_dir=CREMAD_DIR):
    """
    Load CREMA-D dataset.
    Filename format: 1001_DFA_ANG_XX.wav
      field[0] = actor ID (1001–1091) → remapped to 201–291
      field[2] = emotion code (ANG/HAP/NEU/SAD/DIS/FEA)

    Actor IDs are remapped to 201–291 so they never collide with
    RAVDESS actors (1–24) or TESS actors (100–102) in the split logic.
    CREMA-D actors are NEVER in TEST_SPEAKERS so all samples go to train.
    """
    records = []
    files   = glob.glob(os.path.join(data_dir, '**', '*.wav'), recursive=True)
    if not files:
        files = glob.glob(os.path.join(data_dir, '*.wav'))
    if not files:
        print(f"⚠️  No CREMA-D files in {data_dir}")
        print("   Run: python src/download_data.py")
        return pd.DataFrame()

    for fp in files:
        name  = os.path.basename(fp).replace('.wav', '')
        parts = name.split('_')
        if len(parts) < 3:
            continue
        emo_code = parts[2].upper()
        emo      = CREMAD_MAP.get(emo_code)
        if emo is None or emo not in EMOTIONS:
            continue
        try:
            # Remap 1001–1091 → 201–291 to avoid actor ID conflicts
            actor_id = int(parts[0]) - 800
        except ValueError:
            actor_id = 200
        records.append({
            'filepath': fp,
            'emotion':  emo,
            'actor_id': actor_id,
            'dataset':  'cremad',
        })

    df = pd.DataFrame(records)
    if len(df) > 0:
        print(f"✅ CREMA-D: {len(df)} samples, {df['actor_id'].nunique()} actors")
        print(f"   {df['emotion'].value_counts().to_dict()}")
    else:
        print("⚠️  CREMA-D: no usable samples found")
    return df


def load_all_data():
    df_r = load_ravdess()
    df_t = load_tess()
    df_c = load_cremad()

    parts = [p for p in [df_r, df_t, df_c] if len(p) > 0]
    if not parts:
        raise RuntimeError("No data found! Run: python src/download_data.py")

    df = pd.concat(parts, ignore_index=True)
    print(f"\n📊 Total: {len(df)} samples across "
          f"{df['dataset'].nunique()} datasets")
    print(df['emotion'].value_counts().to_dict())
    return df
