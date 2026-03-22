# ================================================================
# src/data_loader.py — Dataset loading for RAVDESS and TESS
# ================================================================

import os, glob
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAVDESS_DIR, TESS_DIR, RAVDESS_MAP, TESS_MAP, EMOTIONS


def load_ravdess(data_dir=RAVDESS_DIR):
    records = []
    files   = glob.glob(os.path.join(data_dir, '**', '*.wav'), recursive=True)
    if not files:
        print(f"⚠️  No RAVDESS files in {data_dir}")
        print("   Run: python src/download_data.py")
        return pd.DataFrame()
    for fp in files:
        parts = os.path.basename(fp).replace('.wav','').split('-')
        if len(parts)==7 and parts[0]=='03' and parts[2] in RAVDESS_MAP:
            records.append({'filepath': fp,
                            'emotion':  RAVDESS_MAP[parts[2]],
                            'actor_id': int(parts[6]),
                            'dataset':  'ravdess'})
    df = pd.DataFrame(records)
    print(f"✅ RAVDESS: {len(df)} samples, {df['actor_id'].nunique()} actors")
    return df


def load_tess(data_dir=TESS_DIR):
    records   = []
    files     = glob.glob(os.path.join(data_dir,'**','*.wav'), recursive=True)
    if not files:
        files = glob.glob(os.path.join(data_dir,'**','*.WAV'), recursive=True)
    if not files:
        print(f"⚠️  No TESS files in {data_dir}")
        return pd.DataFrame()
    actor_map = {'OAF':101,'YAF':102}
    for fp in files:
        parts    = os.path.basename(fp).lower().replace('.wav','').split('_')
        emo      = TESS_MAP.get(parts[-1]) if parts else None
        speaker  = parts[0].upper() if parts else ''
        actor_id = actor_map.get(speaker, 100)
        if emo and emo in EMOTIONS:
            records.append({'filepath': fp, 'emotion': emo,
                            'actor_id': actor_id, 'dataset': 'tess'})
    df = pd.DataFrame(records)
    print(f"✅ TESS: {len(df)} samples")
    return df


def load_all_data():
    df_r = load_ravdess()
    df_t = load_tess()
    parts = [p for p in [df_r, df_t] if len(p)>0]
    if not parts:
        raise RuntimeError("No data! Run: python src/download_data.py")
    df = pd.concat(parts, ignore_index=True)
    print(f"\n📊 Total: {len(df)} samples")
    print(df['emotion'].value_counts().to_dict())
    return df
