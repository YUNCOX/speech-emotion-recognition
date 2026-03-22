# ================================================================
# src/download_data.py — Download RAVDESS and TESS
#
# Usage:
#   python src/download_data.py
# ================================================================

import os, sys, shutil, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAVDESS_DIR, TESS_DIR, DATA_DIR


def check_kaggle():
    kaggle_json = os.path.expanduser('~/.kaggle/kaggle.json')
    if not os.path.exists(kaggle_json):
        print("❌ Kaggle credentials not found!")
        print("   1. Go to kaggle.com → Settings → API → Create New Token")
        print("   2. Place kaggle.json at ~/.kaggle/kaggle.json")
        sys.exit(1)
    print("✅ Kaggle credentials found")


def copy_files(src_path, dst_dir, exts=('.wav','.WAV')):
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
    existing = glob.glob(os.path.join(RAVDESS_DIR, '*.wav'))
    if existing:
        print(f"✅ RAVDESS already present ({len(existing)} files)")
        return
    print("📥 Downloading RAVDESS...")
    import kagglehub
    path = kagglehub.dataset_download(
        'uwrfkaggler/ravdess-emotional-speech-audio')
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


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    check_kaggle()
    download_ravdess()
    download_tess()
    print("\n✅ All datasets ready!")
    print("   Next: python src/train.py")
