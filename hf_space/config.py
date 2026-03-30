# ================================================================
# config.py — Central configuration for Speech Emotion Recognition
# Task #46 | Al-Farabi University | Mohammed Natiq Hilo
# Approach: wav2vec2 embeddings + Statistical MFCC + SVM
# Datasets: RAVDESS + TESS + CREMA-D
# ================================================================

import os

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
RAVDESS_DIR = os.path.join(DATA_DIR, 'ravdess')
TESS_DIR    = os.path.join(DATA_DIR, 'tess')
CREMAD_DIR  = os.path.join(DATA_DIR, 'cremad')
CACHE_FILE  = os.path.join(MODELS_DIR, 'embeddings_cache.pkl')
MODEL_FILE  = os.path.join(MODELS_DIR, 'best_ser_final.pkl')

# ── Audio ─────────────────────────────────────────────────────────
SR_W2V      = 16000      # wav2vec2 requires 16kHz
SR_MFCC     = 22050      # librosa standard
DURATION    = 3.0
N_MFCC      = 40

# ── Emotions ──────────────────────────────────────────────────────
EMOTIONS      = ['angry', 'happy', 'neutral', 'sad']
EMOTION_EMOJI = {'angry':'😠','happy':'😊','neutral':'😐','sad':'😢'}
EMOTION_COLOR = {'angry':'#e74c3c','happy':'#f39c12',
                 'neutral':'#2ecc71','sad':'#3498db'}

RAVDESS_MAP = {'01':'neutral','03':'happy','04':'sad','05':'angry'}
TESS_MAP    = {'angry':'angry','happy':'happy','sad':'sad',
               'neutral':'neutral','ps':'happy'}
CREMAD_MAP  = {'ANG':'angry','HAP':'happy','NEU':'neutral','SAD':'sad'}

# ── Model ─────────────────────────────────────────────────────────
WAV2VEC_MODEL = "facebook/wav2vec2-base"
BATCH_SIZE    = 64
TEST_SPEAKERS = {1,9,12,17,19}
SVM_C_VALUES  = [0.5,1,5,10,50]
TARGET_F1     = 0.72
N_AUG_NEUTRAL = 5
N_AUG_OTHER   = 3

# ── Ensure dirs exist ─────────────────────────────────────────────
for d in [DATA_DIR, RAVDESS_DIR, TESS_DIR, CREMAD_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)
