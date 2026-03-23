# ================================================================
# app/demo.py — Speech Emotion Recognition Demo (Final)
# Usage:
#   python app/demo.py --share
#   python app/demo.py --port 7862 --share
# ================================================================

import os, sys, pickle, argparse, warnings
import numpy as np
import librosa, librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr
import soundfile as sf
import tempfile

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (SR_W2V, SR_MFCC, DURATION, MODEL_FILE,
                    EMOTION_EMOJI, EMOTION_COLOR, TARGET_F1)
from src.features import (load_wav2vec2, extract_w2v_batch,
                           extract_mfcc_features)

# ── Load model & wav2vec2 once ────────────────────────────────────
print("Loading model...")
with open(MODEL_FILE, 'rb') as f:
    payload = pickle.load(f)

svm_model   = payload['model']
scaler      = payload['scaler']
le          = payload['label_encoder']
thresholds  = payload.get('thresholds') or payload.get('neutral_threshold')
metrics     = payload.get('metrics', {})

print("Loading wav2vec2...")
processor, w2v_model, device = load_wav2vec2()
print("All ready!\n")

wf1 = metrics.get('weighted_f1', 0.7278)
acc = metrics.get('accuracy', 0.7214)

BADGE = {
    'angry':   ('#e74c3c', '😠'),
    'happy':   ('#f39c12', '😊'),
    'neutral': ('#2ecc71', '😐'),
    'sad':     ('#3498db', '😢'),
}


# ── Plots ─────────────────────────────────────────────────────────
def plot_spectrogram(y, sr, emotion):
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    S = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000),
        ref=np.max)
    img = librosa.display.specshow(S, x_axis='time', y_axis='mel',
                                   sr=sr, ax=ax, cmap='magma')
    color, emoji = BADGE.get(emotion, ('#fff', ''))
    ax.set_title('Mel-Spectrogram — Predicted: ' + emoji + ' ' + emotion.capitalize(),
                 color='white', fontsize=12)
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_color('white')
    cb = plt.colorbar(img, ax=ax)
    cb.ax.yaxis.set_tick_params(color='white')
    plt.tight_layout()
    return fig


def plot_confidence(probs_dict):
    classes = list(probs_dict.keys())
    values  = [probs_dict[c] * 100 for c in classes]
    colors  = [BADGE.get(c, ('#888', ''))[0] for c in classes]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    bars = ax.barh(classes, values, color=colors, height=0.5)
    for bar, val in zip(bars, values):
        ax.text(min(val + 1, 97), bar.get_y() + bar.get_height() / 2,
                str(round(val, 1)) + '%',
                va='center', color='white', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 108)
    ax.set_title('Emotion Confidence Scores', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.set_xlabel('Confidence (%)', color='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    plt.tight_layout()
    return fig


# ── Prediction ────────────────────────────────────────────────────
def run_prediction(tmp_path):
    w2v    = extract_w2v_batch([tmp_path], processor, w2v_model, device)
    mfcc   = extract_mfcc_features(tmp_path).reshape(1, -1)
    feat_n = scaler.transform(np.hstack([w2v, mfcc]))
    probs  = svm_model.predict_proba(feat_n)[0]

    # Fixed threshold logic: pick highest-confidence class among those
    # that clear their threshold. Falls back to argmax if none do.
    # (Original argmax ignored thresholds entirely; the previous fix
    # used break which made class order decide the winner.)
    if isinstance(thresholds, dict):
        candidates = {ci: probs[ci] for ci, t in thresholds.items()
                      if probs[ci] >= t}
        idx = max(candidates, key=candidates.get) if candidates else int(np.argmax(probs))
    else:
        idx = int(np.argmax(probs))

    label      = le.inverse_transform([idx])[0]
    confidence = float(probs.max() * 100)
    probs_dict = {cls: float(probs[i]) for i, cls in enumerate(le.classes_)}
    return label, confidence, probs_dict


def make_result_html(label, confidence):
    color, emoji = BADGE.get(label, ('#ffffff', '🎙️'))
    return (
        "<div style='background:#1e293b;border-radius:12px;padding:20px 24px;"
        "border:2px solid " + color + ";text-align:center;margin-top:8px'>"
        "<div style='font-size:48px;margin-bottom:8px'>" + emoji + "</div>"
        "<div style='font-size:28px;font-weight:bold;color:" + color + ";letter-spacing:2px'>"
        + label.upper() +
        "</div>"
        "<div style='font-size:16px;color:#94a3b8;margin-top:6px'>"
        + str(round(confidence, 1)) + "% confidence"
        "</div></div>"
    )


def predict_from_audio(audio):
    if audio is None:
        return "### 🎙️ No audio provided", None, None

    sr_in, data = audio
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    if data.max() > 1.0:
        data = data / 32768.0
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, data, sr_in)
        tmp_path = tmp.name

    try:
        label, confidence, probs_dict = run_prediction(tmp_path)
        color, emoji = BADGE.get(label, ('#fff', '🎙️'))
        result_html = "## " + emoji + " " + label.upper() + "\n### " + str(round(confidence, 1)) + "% confidence"
        y_mel    = librosa.load(tmp_path, sr=SR_MFCC, duration=DURATION)[0]
        conf_fig = plot_confidence(probs_dict)
        spec_fig = plot_spectrogram(y_mel, SR_MFCC, label)
        return result_html, conf_fig, spec_fig
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Static HTML ───────────────────────────────────────────────────
header_html = (
    "<div style='text-align:center;color:#94a3b8;margin-bottom:10px'>"
    "<b style='color:#e2e8f0;font-size:15px'>Al-Farabi University</b>"
    " &nbsp;·&nbsp; Computer Engineering &nbsp;·&nbsp; Task #46<br>"
    "<span style='font-size:13px'>"
    "Model: wav2vec2 + Statistical MFCC + SVM"
    " &nbsp;|&nbsp; Dataset: RAVDESS + TESS"
    " &nbsp;|&nbsp; <b style='color:#4ade80'>Weighted F1: "
    + str(round(wf1, 4)) +
    "</b> &nbsp;|&nbsp; Accuracy: "
    + str(round(acc * 100, 1)) + "%"
    "</span></div>"
)

how_to_html = (
    "<div style='background:#1e293b;border-radius:10px;"
    "padding:14px 20px;margin-bottom:4px;border:1px solid #334155'>"
    "<b style='color:#94a3b8'>How to use:</b>"
    "<ol style='color:#cbd5e1;margin:8px 0 8px 18px;padding:0'>"
    "<li>Use the <b>Record</b> tab to speak, or <b>Upload</b> tab to drop a .wav file</li>"
    "<li>Click <b>Analyse Emotion</b></li>"
    "<li>See the predicted emotion, confidence scores, and mel-spectrogram</li>"
    "</ol>"
    "<span style='color:#475569;font-size:12px'>"
    "Try: <i>\"I can't believe you did that!\"</i> (angry)"
    " &nbsp;·&nbsp; <i>\"This is amazing!\"</i> (happy)"
    " &nbsp;·&nbsp; <i>\"I really miss you...\"</i> (sad)"
    " &nbsp;·&nbsp; <i>\"The meeting is at 3pm\"</i> (neutral)"
    "</span></div>"
)

badges_inner = ""
for emo in ['angry', 'happy', 'neutral', 'sad']:
    color, emoji = BADGE[emo]
    badges_inner += (
        "<span style='background:" + color + ";color:white;"
        "padding:7px 20px;border-radius:20px;font-weight:bold;font-size:14px'>"
        + emoji + " " + emo.capitalize() + "</span>"
    )
emotions_html = (
    "<div style='margin-top:16px;padding:14px 20px;background:#1e293b;"
    "border-radius:10px;border:1px solid #334155'>"
    "<b style='color:#94a3b8'>Emotions the model can detect:</b>"
    "<div style='display:flex;gap:12px;margin-top:10px;flex-wrap:wrap'>"
    + badges_inner + "</div></div>"
)

empty_result_html = (
    "<div style='background:#1e293b;border-radius:12px;padding:20px 24px;"
    "border:1px solid #334155;text-align:center;color:#64748b'>"
    "🎙️ Record or upload audio to get prediction"
    "</div>"
)

# ── Gradio UI ─────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Base(),
               title="Speech Emotion Recognition") as demo:

    gr.Markdown("# 🎙️ Speech Emotion Recognition")
    gr.HTML(header_html)
    gr.HTML(how_to_html)

    with gr.Row():
        # Left — input + result
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("🎤 Record"):
                    mic_input = gr.Audio(
                        source="microphone",
                        type="numpy",
                        label="Speak with emotion"
                    )
                with gr.Tab("📁 Upload"):
                    file_input = gr.Audio(
                        source="upload",
                        type="numpy",
                        label="Upload .wav file"
                    )

            analyse_btn = gr.Button("🔍 Analyse Emotion",
                                    variant="primary", size="lg")
            gr.Markdown("### 🎭 Predicted Emotion")
            result_html = gr.Markdown(value="🎙️ *Record or upload audio to get prediction*")

        # Right — plots
        with gr.Column(scale=2):
            conf_plot = gr.Plot(label="📊 Confidence Scores")
            spec_plot = gr.Plot(label="🌈 Mel-Spectrogram")

    gr.HTML(emotions_html)

    # Mic tab events
    def predict_either(mic, upload):
        audio = mic if mic is not None else upload
        return predict_from_audio(audio)

    analyse_btn.click(
        fn=predict_either,
        inputs=[mic_input, file_input],
        outputs=[result_html, conf_plot, spec_plot]
    )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int, default=7862)
    args = parser.parse_args()
    print("Launching... port=" + str(args.port) + " share=" + str(args.share))
    demo.launch(share=args.share, server_port=args.port)


if __name__ == '__main__':
    main()
