# 🎙️ Speech Emotion Recognition
### Task #46 — Al-Farabi University | Department of Computer Engineering

> **Author:** Mohammed Natiq Hilo
> **Approach:** wav2vec2 embeddings + Statistical MFCC + SVM
> **Target:** Weighted F1-score > 0.72 on held-out speaker test set

---

## 🎯 Live Demo

**👉 [Try the Live Demo](https://c935ae52930645ac98.gradio.live)**

Record your voice or upload a .wav file and see the emotion predicted in real time with confidence scores and mel-spectrogram visualization.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Weighted F1** | **0.7278** ✅ |
| Accuracy | 72.14% |
| Macro F1 | 0.7289 |

**Per-class F1 scores:**

| Emotion | F1 Score | ≥ 0.72 |
|---------|----------|--------|
| 😠 Angry   | 0.8451 | ✅ |
| 😊 Happy   | 0.6591 | ❌ |
| 😐 Neutral | 0.7368 | ✅ |
| 😢 Sad     | 0.6747 | ❌ |

> Weighted F1 target (> 0.72) achieved. See Critical Analysis for per-class discussion.

---

## 🏗️ Architecture

```
Audio (.wav)
    ↓
wav2vec2-base (768-dim) ──┐
                           ├── Concatenate → 1805-dim vector
Statistical MFCC (1037-dim)┘
    ↓
StandardScaler normalization
    ↓
Smart Augmentation (3-5x, neutral-weighted)
    ↓
SVM RBF (C=1, grid search) + Per-class threshold calibration
    ↓
Emotion: angry / happy / neutral / sad
```

**Why this combination?**
- **wav2vec2** captures deep prosodic and phonetic patterns from 960h pre-training
- **Statistical MFCC** provides interpretable acoustic features (energy, pitch, timbre)
- **Per-class calibration** optimises each emotion's decision threshold independently
- **Smart augmentation** balances the neutral class imbalance (5x vs 3x for others)

---

## 📁 Project Structure

```
speech-emotion-recognition/
├── config.py                 ← all settings
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── download_data.py      ← download RAVDESS + TESS
│   ├── data_loader.py        ← parse datasets
│   ├── features.py           ← wav2vec2 + MFCC extraction
│   ├── train.py              ← full training pipeline
│   └── predict.py            ← CLI inference
├── app/
│   └── demo.py               ← Gradio live demo
├── notebooks/
│   └── speech_emotion_recognition.ipynb
└── results/
    ├── results.png
    └── metrics.json
```

---

## 🚀 Quick Start

### 1. Install PyTorch with GPU support (NVIDIA RTX recommended)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install remaining requirements
```bash
pip install -r requirements.txt
```

### 3. Verify GPU
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 4. Download datasets
```bash
python src/download_data.py
```

### 5. Train (~5 min on GPU / ~60 min on CPU)
```bash
python src/train.py
```
Features are cached after first run — re-runs are instant.

### 6. Run live demo
```bash
python app/demo.py --share    # public Gradio link
python app/demo.py            # local only (port 7862)
```

### 7. Predict a single audio file
```bash
python src/predict.py --file path/to/audio.wav
```

---

## 🔬 Experimental Progression

| Version | Approach | Weighted F1 |
|---------|----------|-------------|
| v1 | Deep CNN (2D MFCC maps) | 0.13 |
| v2 | CNN + Data Augmentation | 0.28 |
| v3 | Statistical MFCC + MLP | 0.66 |
| v4 | Statistical MFCC + SVM | 0.61 |
| v5 | SVM + TESS (wrong split) | 0.48 |
| v6 | SVM + corrected split | 0.6579 |
| v7 | XGBoost + Pitch features | 0.61 |
| v8 | Rich MFCC + SVM + calibration | 0.6814 |
| **Final** | **wav2vec2 + MFCC + SVM + GPU** | **0.7278 ✅** |

---

## 📦 Datasets

| Dataset | Speakers | Samples Used | Source |
|---------|----------|-------------|--------|
| RAVDESS | 24 actors (12M, 12F) | 672 | [Zenodo](https://zenodo.org/record/1188976) |
| TESS | 2 female speakers | 2,000 | [Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) |

**Evaluation protocol:** Speaker-independent — RAVDESS actors 1, 9, 12, 17, 19 held out exclusively for testing. All TESS samples remain in training only.

---

## 🔍 Critical Analysis

**Why the weighted F1 target was achieved but not all per-class scores:**

1. **Small speaker pool** — only 5 test speakers from RAVDESS; one atypical speaker significantly shifts per-class metrics
2. **Neutral class imbalance** — only 20 neutral test samples vs 40 for other classes
3. **Cross-dataset domain mismatch** — TESS (clean studio) vs RAVDESS (acted speech) creates distribution shift
4. **Speaker-independent evaluation** — most published results exceeding F1=0.72 per class use speaker-dependent splits

The angry class (F1=0.8451) and neutral class (F1=0.7368) both exceed the target. Happy and sad are acoustically similar — both exhibit moderate energy and pitch patterns — causing frequent confusion between them.

---

## 📹 Video Presentation

**YouTube:** [Link to be added after recording]

---

## 📄 Technical Report

Full IEEE-format technical report (4-6 pages) available in the repository:
`SER_Paper_Mohammed_Natiq_Hilo.docx`

---

## 📚 References

1. A. Baevski et al., "wav2vec 2.0: A framework for self-supervised learning of speech representations," NeurIPS, 2020.
2. S. R. Livingstone and F. A. Russo, "The RAVDESS dataset," PLOS ONE, 2018.
3. K. Dupuis and M. K. Pichora-Fuller, "TESS dataset," University of Toronto, 2010.
4. B. Schuller et al., "The INTERSPEECH 2009 emotion challenge," Proc. Interspeech, 2009.

---

## 📜 License

Academic coursework — Al-Farabi University College, Department of Computer Engineering.
RAVDESS used under CC BY-NC-SA 4.0.
