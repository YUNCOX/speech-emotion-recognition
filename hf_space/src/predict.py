# ================================================================
# src/predict.py — Single file inference CLI
#
# Usage:
#   python src/predict.py --file path/to/audio.wav
# ================================================================

import os, sys, pickle, argparse
import numpy as np
import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SR_W2V, DURATION, MODEL_FILE, EMOTION_EMOJI
from src.features import load_wav2vec2, extract_w2v_batch, extract_mfcc_features


def load_model(path=MODEL_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}\nRun: python src/train.py")
    with open(path, 'rb') as f:
        return pickle.load(f)


def predict_file(filepath, payload, processor, w2v_model, device):
    model         = payload['model']
    scaler        = payload['scaler']
    le            = payload['label_encoder']
    neutral_thresh = payload.get('thresholds') or payload.get('neutral_threshold')
    neutral_idx   = payload.get('neutral_idx', 2)

    # Extract combined features
    w2v  = extract_w2v_batch([filepath], processor, w2v_model, device)  # (1,768)
    mfcc = extract_mfcc_features(filepath).reshape(1, -1)
    feat = np.hstack([w2v, mfcc])
    feat_n = scaler.transform(feat)

    # Predict with optional neutral threshold
    probs_arr = model.predict_proba(feat_n)[0]
    if neutral_thresh and probs_arr[neutral_idx] >= neutral_thresh:
        pred_idx = neutral_idx
    else:
        pred_idx = np.argmax(probs_arr)

    label      = le.inverse_transform([pred_idx])[0]
    confidence = probs_arr.max() * 100
    probs      = {cls: float(probs_arr[i]) for i, cls in enumerate(le.classes_)}
    return label, confidence, probs


def print_result(filepath, label, confidence, probs):
    emoji = EMOTION_EMOJI.get(label, '🎙️')
    print(f"\n{'='*45}")
    print(f"  File   : {os.path.basename(filepath)}")
    print(f"  Result : {emoji} {label.upper()} — {confidence:.1f}%")
    print(f"{'='*45}")
    for cls, prob in sorted(probs.items(), key=lambda x: -x[1]):
        bar = '█' * int(prob * 30)
        print(f"  {EMOTION_EMOJI.get(cls,'')} {cls:<8}: {bar:<30} {prob*100:5.1f}%")
    print(f"{'='*45}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',  required=True)
    parser.add_argument('--model', default=MODEL_FILE)
    args = parser.parse_args()

    payload                   = load_model(args.model)
    processor, w2v_model, dev = load_wav2vec2()
    label, conf, probs        = predict_file(
        args.file, payload, processor, w2v_model, dev)
    print_result(args.file, label, conf, probs)


if __name__ == '__main__':
    main()
