# ================================================================
# src/features.py — Combined wav2vec2 + Statistical MFCC features
#
# Strategy: concatenate wav2vec2 (768-dim) + rich MFCC stats (1043-dim)
# = 1811-dim vector → SVM
# This combines the deep representations of wav2vec2 with the
# interpretable acoustic features of v8, giving the best of both.
# ================================================================

import os, pickle
import numpy as np
import librosa
import torch
from tqdm import tqdm
from scipy import stats as scipy_stats
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (WAV2VEC_MODEL, SR_W2V, SR_MFCC, DURATION,
                    N_MFCC, BATCH_SIZE, CACHE_FILE)


def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"🎮 GPU detected: {name}")
        return 'cuda'
    print("💻 No GPU detected — using CPU")
    return 'cpu'


def load_wav2vec2():
    print(f"📥 Loading {WAV2VEC_MODEL}...")
    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL)
    model     = Wav2Vec2Model.from_pretrained(
                    WAV2VEC_MODEL,
                    use_safetensors=True)   # ← avoids torch.load entirely
    device    = get_device()
    model     = model.to(device).eval()
    print(f"✅ wav2vec2 ready on {device}")
    return processor, model, device


def load_audio_w2v(filepath):
    """Load audio at 16kHz for wav2vec2."""
    target = int(SR_W2V * DURATION)
    y, _   = librosa.load(filepath, sr=SR_W2V, duration=DURATION, mono=True)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    return y[:target].astype(np.float32)


def extract_w2v_batch(filepaths, processor, model, device):
    """Extract wav2vec2 embeddings for a batch — returns (B, 768)."""
    waves = []
    for fp in filepaths:
        try:
            waves.append(load_audio_w2v(fp))
        except:
            waves.append(np.zeros(int(SR_W2V*DURATION), dtype=np.float32))

    inputs = processor(waves, sampling_rate=SR_W2V,
                       return_tensors="pt", padding=True)
    iv = inputs.input_values.to(device)
    with torch.no_grad():
        out = model(iv).last_hidden_state.mean(dim=1)  # (B, 768)
    return out.cpu().numpy()


def extract_mfcc_features(filepath):
    """
    Extract rich statistical MFCC features (same as v8 approach).
    Returns ~1043-dim vector.
    """
    y, _ = librosa.load(filepath, sr=SR_MFCC, duration=DURATION, mono=True)
    target = int(SR_MFCC * DURATION)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]

    feats = []

    # 1. MFCC + delta + delta2 with 6 stats each → 3×40×6 = 720
    mfcc    = librosa.feature.mfcc(y=y, sr=SR_MFCC, n_mfcc=N_MFCC)
    mfcc_d  = librosa.feature.delta(mfcc)
    mfcc_d2 = librosa.feature.delta(mfcc, order=2)
    for m in [mfcc, mfcc_d, mfcc_d2]:
        feats.extend([
            m.mean(axis=1), m.std(axis=1),
            m.min(axis=1),  m.max(axis=1),
            scipy_stats.skew(m, axis=1),
            scipy_stats.kurtosis(m, axis=1),
        ])

    # 2. Mel-Spectrogram → 128×2 = 256
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=SR_MFCC, n_mels=128, fmax=8000),
        ref=np.max)
    feats.extend([mel.mean(axis=1), mel.std(axis=1)])

    # 3. Chroma → 12×2 = 24
    chroma = librosa.feature.chroma_stft(y=y, sr=SR_MFCC)
    feats.extend([chroma.mean(axis=1), chroma.std(axis=1)])

    # 4. Spectral contrast → 7×2 = 14
    contrast = librosa.feature.spectral_contrast(y=y, sr=SR_MFCC)
    feats.extend([contrast.mean(axis=1), contrast.std(axis=1)])

    # 5. Energy trajectory → 12 dims
    rms  = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    n3   = max(1, len(rms)//3)
    s1,s2,s3 = rms[:n3].mean(), rms[n3:2*n3].mean(), rms[2*n3:].mean()
    x    = np.arange(len(rms))
    slope = np.polyfit(x, rms, 1)[0]
    feats.append(np.array([
        rms.mean(), rms.std(), rms.max(), rms.min(),
        s1, s2, s3, s3-s1,
        np.argmax(rms)/len(rms), slope,
        (rms > rms.mean()).mean(),
        np.percentile(rms,90) - np.percentile(rms,10),
    ]))

    # 6. Spectral shape → 6 dims
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=SR_MFCC)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=SR_MFCC)[0]
    bwidth   = librosa.feature.spectral_bandwidth(y=y, sr=SR_MFCC)[0]
    feats.append(np.array([
        rolloff.mean(), rolloff.std(),
        centroid.mean(), centroid.std(),
        bwidth.mean(), bwidth.std(),
    ]))

    # 7. ZCR + flux → 5 dims
    zcr  = librosa.feature.zero_crossing_rate(y)[0]
    stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
    flux = np.sqrt(np.mean(np.diff(stft, axis=1)**2, axis=0))
    feats.append(np.array([
        zcr.mean(), zcr.std(),
        flux.mean(), flux.std(), flux.max(),
    ]))

    return np.concatenate(feats)


def extract_all_features(df, processor, model, device,
                         cache_file=CACHE_FILE, force=False):
    """
    Extract combined wav2vec2 + MFCC features for all files.
    Caches to disk — re-runs are instant.
    """
    if not force and os.path.exists(cache_file):
        print(f"⚡ Loading cached features from {cache_file}...")
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        print(f"✅ Loaded {len(cache['X'])} samples instantly!")
        return cache['X'], cache['y'], cache['g']

    filepaths = df['filepath'].tolist()
    emotions  = df['emotion'].tolist()
    actors    = df['actor_id'].tolist()
    N         = len(filepaths)

    print(f"\n⚙️  Extracting wav2vec2 embeddings (batch={BATCH_SIZE})...")
    w2v_feats = []
    for i in tqdm(range(0, N, BATCH_SIZE), desc="wav2vec2"):
        batch = filepaths[i:i+BATCH_SIZE]
        embs  = extract_w2v_batch(batch, processor, model, device)
        w2v_feats.append(embs)
    W = np.vstack(w2v_feats)   # (N, 768)
    print(f"✅ wav2vec2 embeddings: {W.shape}")

    print(f"\n⚙️  Extracting statistical MFCC features...")
    mfcc_feats = []
    for fp in tqdm(filepaths, desc="MFCC"):
        try:
            mfcc_feats.append(extract_mfcc_features(fp))
        except:
            mfcc_feats.append(np.zeros(1037, dtype=np.float32))
    M = np.array(mfcc_feats)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"✅ MFCC features: {M.shape}")

    # Concatenate → (N, 768 + MFCC_dims)
    X = np.hstack([W, M])
    print(f"✅ Combined features: {X.shape}")

    y = emotions
    g = np.array(actors)

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump({'X': X, 'y': y, 'g': g}, f)
    print(f"✅ Cached to: {cache_file}")
    return X, y, g
