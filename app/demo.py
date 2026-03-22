# ================================================================
# app/demo.py — Gradio Live Demo
# Usage:
#   python app/demo.py --share      # public link
#   python app/demo.py              # local only
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
print("📥 Loading model...")
with open(MODEL_FILE, 'rb') as f:
    payload = pickle.load(f)

svm_model      = payload['model']
scaler         = payload['scaler']
le             = payload['label_encoder']
thresholds     = payload.get('thresholds') or payload.get('neutral_threshold')
neutral_idx    = payload.get('neutral_idx', 2)
metrics        = payload.get('metrics', {})

print("📥 Loading wav2vec2...")
processor, w2v_model, device = load_wav2vec2()
print("✅ All ready!\n")


def plot_spectrogram(y, sr, emotion):
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    S = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000),
        ref=np.max)
    img = librosa.display.specshow(S, x_axis='time', y_axis='mel',
                                   sr=sr, ax=ax, cmap='magma')
    emoji = EMOTION_EMOJI.get(emotion, '')
    ax.set_title('Mel-Spectrogram — ' + emoji + ' ' + emotion.capitalize(),
                 color='white', fontsize=12)
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_color('white')
    plt.colorbar(img, ax=ax).ax.yaxis.set_tick_params(color='white')
    plt.tight_layout()
    return fig


def plot_confidence(probs_dict, emotion):
    classes = list(probs_dict.keys())
    values  = [probs_dict[c] * 100 for c in classes]
    colors  = [EMOTION_COLOR.get(c, '#888') for c in classes]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    bars = ax.barh(classes, values, color=colors, height=0.5)
    for bar, val in zip(bars, values):
        ax.text(min(val + 1, 98), bar.get_y() + bar.get_height() / 2,
                str(round(val, 1)) + '%', va='center', color='white',
                fontsize=11, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.set_title('Confidence Scores', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    plt.tight_layout()
    return fig


def predict_emotion(audio):
    if audio is None:
        return "⚠️ No audio provided", None, None

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
        w2v  = extract_w2v_batch([tmp_path], processor, w2v_model, device)
        mfcc = extract_mfcc_features(tmp_path).reshape(1, -1)
        feat = np.hstack([w2v, mfcc])
        feat_n = scaler.transform(feat)

        probs_arr = svm_model.predict_proba(feat_n)[0]

        # Apply thresholds
        if isinstance(thresholds, dict):
            pred_idx = np.argmax(probs_arr)
            for cls_idx, thresh in thresholds.items():
                if probs_arr[cls_idx] >= thresh:
                    pred_idx = cls_idx
                    break
        else:
            pred_idx = np.argmax(probs_arr)

        label      = le.inverse_transform([pred_idx])[0]
        confidence = probs_arr.max() * 100
        probs_dict = {}
        for i, cls in enumerate(le.classes_):
            probs_dict[cls] = float(probs_arr[i])

        emoji       = EMOTION_EMOJI.get(label, '')
        result_text = "## " + emoji + " " + label.upper() + " — " + str(round(confidence, 1)) + "% confidence"

        y_mel    = librosa.load(tmp_path, sr=SR_MFCC, duration=DURATION)[0]
        conf_fig = plot_confidence(probs_dict, label)
        spec_fig = plot_spectrogram(y_mel, SR_MFCC, label)
        return result_text, conf_fig, spec_fig

    finally:
        os.unlink(tmp_path)


# ── Build UI ──────────────────────────────────────────────────────
wf1  = metrics.get('weighted_f1', 0)
acc  = metrics.get('accuracy', 0)
pcf1 = metrics.get('per_class_f1', {})

desc = (
    "<div style='text-align:center;color:#aaa;margin-bottom:12px'>"
    "<b>Al-Farabi University</b> · Computer Engineering · Task #46<br>"
    "Model: wav2vec2 + Statistical MFCC + SVM &nbsp;|&nbsp;"
    "Dataset: RAVDESS + TESS &nbsp;|&nbsp;"
    "Weighted F1: " + str(round(wf1, 4)) + " &nbsp;|&nbsp;"
    "Accuracy: " + str(round(acc * 100, 1)) + "%"
    "</div>"
)

how_to = (
    "**How to use:**\n"
    "1. Click **Record** and speak with emotion (2-3 seconds)\n"
    "2. Click **Analyse Emotion**\n"
    "3. See prediction + confidence + mel-spectrogram\n\n"
    "Try: *'I can't believe you did that!'* (angry) · "
    "*'This is amazing!'* (happy) · "
    "*'I really miss you...'* (sad) · "
    "*'The meeting is at 3pm'* (neutral)"
)

# Build per-class table HTML
table_html = ""
if pcf1:
    rows = ""
    for c in sorted(pcf1.keys()):
        f      = pcf1[c]
        emoji  = EMOTION_EMOJI.get(c, '')
        color  = EMOTION_COLOR.get(c, '#fff')
        status = "✅" if f >= TARGET_F1 else "❌"
        rows += (
            "<tr>"
            "<td>" + emoji + " " + c.capitalize() + "</td>"
            "<td style='color:" + color + "'>" + str(round(f, 4)) + "</td>"
            "<td>" + status + "</td>"
            "</tr>"
        )
    table_html = (
        "<div style='margin-top:16px;padding:12px 20px;background:#1e293b;"
        "border-radius:10px;border:1px solid #334155'>"
        "<b style='color:#94a3b8'>Per-Class F1 (Test Set):</b>"
        "<table style='width:100%;margin-top:8px;color:white;border-collapse:collapse'>"
        "<tr style='color:#94a3b8'>"
        "<th align='left'>Emotion</th>"
        "<th align='left'>F1</th>"
        "<th align='left'>Target</th>"
        "</tr>"
        + rows +
        "</table></div>"
    )

with gr.Blocks(theme=gr.themes.Base(),
               title="Speech Emotion Recognition") as demo:
    gr.Markdown("# 🎙️ Speech Emotion Recognition")
    gr.HTML(desc)
    gr.Markdown(how_to)

    with gr.Row():
        with gr.Column(scale=1):
            audio_in    = gr.Audio(sources=["microphone", "upload"],
                                   type="numpy",
                                   label="🎤 Record or Upload")
            analyse_btn = gr.Button("🔍 Analyse Emotion",
                                    variant="primary", size="lg")
            result_out  = gr.Markdown(label="🎭 Predicted Emotion")
        with gr.Column(scale=2):
            conf_plot = gr.Plot(label="📊 Confidence Scores")
            spec_plot = gr.Plot(label="🌈 Mel-Spectrogram")

    if table_html:
        gr.HTML(table_html)

    analyse_btn.click(fn=predict_emotion,
                      inputs=[audio_in],
                      outputs=[result_out, conf_plot, spec_plot])
    audio_in.change(fn=predict_emotion,
                    inputs=[audio_in],
                    outputs=[result_out, conf_plot, spec_plot])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    print("🚀 Launching (share=" + str(args.share) + ", port=" + str(args.port) + ")...")
    demo.launch(share=args.share, server_port=args.port)


if __name__ == '__main__':
    main()