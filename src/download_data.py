# ================================================================
# src/download_data.py — Download RAVDESS (full), TESS, and CREMA-D
#
# Datasets:
#   RAVDESS full : 2452 files (speech + song, 24 actors)
#                  Speech only: actor-01 to actor-24, modality 03
#                  Song only:   actor-01 to actor-24, modality 04
#   TESS         : ~2800 files (2 female speakers)
#   CREMA-D      : 7442 files (91 actors, diverse demographics)
#
# Combined training data (after speaker-independent split):
#   ~10,000+ samples → significantly improves happy and sad F1
#
# Usage:
#   python src/download_data.py
# ================================================================

import os, sys, shutil, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAVDESS_DIR, TESS_DIR, CREMAD_DIR, DATA_DIR


def check_kaggle():
    kaggle_json = os.path.expanduser('~/.kaggle/kaggle.json')
    if not os.path.exists(kaggle_json):
        print("❌ Kaggle credentials not found!")
        print("   1. Go to kaggle.com → Settings → API → Create New Token")
        print("   2. Place kaggle.json at ~/.kaggle/kaggle.json")
        sys.exit(1)
    print("✅ Kaggle credentials found")


def copy_files(src_path, dst_dir, exts=('.wav', '.WAV')):
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    for root, _, files in os.walk(src_path):
        for f in files:
            if any(f.endswith(e) for e in exts):
                shutil.copy2(os.path.join(root, f),
                             os.path.join(dst_dir, f))
                count += 1
    return count


def download_ravdess():
    """
    Download FULL RAVDESS (speech + song).
    The speech-only dataset ('uwrfkaggler/ravdess-emotional-speech-audio')
    has 1440 files but uses a different modality prefix.
    The full dataset has 2452 files across both modalities.

    Both modality 03 (speech) and modality 04 (song) use the same
    emotion codes, so data_loader.py filters by modality field.
    """
    existing = glob.glob(os.path.join(RAVDESS_DIR, '*.wav'))
    if len(existing) >= 1344:
        print(f"✅ RAVDESS already present ({len(existing)} files)")
        return
    if existing:
        print(f"⚠️  Only {len(existing)} RAVDESS files found — re-downloading full dataset...")

    print("📥 Downloading RAVDESS (speech + song)...")
    import kagglehub
    # Full RAVDESS dataset — speech AND song
    path = kagglehub.dataset_download(
        'uwrfkaggler/ravdess-emotional-speech-and-song')
    n = copy_files(path, RAVDESS_DIR)
    print(f"✅ RAVDESS: {n} files → {RAVDESS_DIR}")


def download_tess():
    existing = glob.glob(os.path.join(TESS_DIR, '**', '*.wav'),
                         recursive=True)
    if existing:
        print(f"✅ TESS already present ({len(existing)} files)")
        return
    print("📥 Downloading TESS...")
    import kagglehub
    path = kagglehub.dataset_download(
        'ejlok1/toronto-emotional-speech-set-tess')
    n = copy_files(path, TESS_DIR)
    print(f"✅ TESS: {n} files → {TESS_DIR}")


def download_cremad():
    """
    Download CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset).
    7,442 clips from 91 actors (ages 20-74, diverse ethnicities).
    Emotions: ANG, DIS, FEA, HAP, NEU, SAD — we use ANG/HAP/NEU/SAD.
    Filename format: 1001_DFA_ANG_XX.wav

    This dataset specifically strengthens happy and sad recognition
    by providing ~1,200+ clips per emotion from diverse speakers.
    """
    existing = glob.glob(os.path.join(CREMAD_DIR, '*.wav'))
    if len(existing) >= 100:
        print(f"✅ CREMA-D already present ({len(existing)} files)")
        return
    print("📥 Downloading CREMA-D (~7,400 files, this may take a few minutes)...")
    import kagglehub
    path = kagglehub.dataset_download('ejlok1/cremad')
    n = copy_files(path, CREMAD_DIR)
    print(f"✅ CREMA-D: {n} files → {CREMAD_DIR}")


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    check_kaggle()
    download_ravdess()
    download_tess()
    download_cremad()

    # Summary
    ravdess_n = len(glob.glob(os.path.join(RAVDESS_DIR, '*.wav')))
    tess_n    = len(glob.glob(os.path.join(TESS_DIR, '**', '*.wav'), recursive=True))
    cremad_n  = len(glob.glob(os.path.join(CREMAD_DIR, '*.wav')))
    print(f"\n✅ All datasets ready!")
    print(f"   RAVDESS : {ravdess_n} files")
    print(f"   TESS    : {tess_n} files")
    print(f"   CREMA-D : {cremad_n} files")
    print(f"   Next: python src/train.py --force")
