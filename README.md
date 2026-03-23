# 🎙️ Speech Emotion Recognition
### Task #46 — Al-Farabi University | Department of Computer Engineering

> **Author:** Mohammed Natiq Hilo
> **Approach:** wav2vec2 embeddings + Statistical MFCC + SVM
> **Datasets:** RAVDESS + TESS + CREMA-D
> **Target:** Weighted F1-score > 0.72 on held-out speaker test set

---

## 🎯 Live Demo

**👉 [Try the Live Demo](DEMO_LINK_HERE)**

Record your voice or upload a .wav file and see the emotion predicted in real time with confidence scores and mel-spectrogram visualization.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Weighted F1** | **0.7711** ✅ |
| Accuracy | 76.79% |
| Macro F1 | 0.7740 |

**Per-class F1 scores:**

| Emotion | F1 Score | ≥ 0.72 |
|---------|----------|--------|
| 😠 Angry   | 0.8267 | ✅ |
| 😊 Happy   | 0.7451 | ✅ |
| 😐 Neutral | 0.7941 | ✅ |
| 😢 Sad     | 0.7302 | ✅ |

> All 4 emotions exceed the target F1 ≥ 0.72. Evaluated on a speaker-independent held-out test set of 280 samples (RAVDESS actors 1, 9, 12, 17, 19).

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
Smart Augmentation (neutral + sad: 5x, others: 3x)
    ↓
SVM RBF (C=0.5, grid search over [0.5, 1, 5, 10, 50])
    ↓
Emotion: angry / happy / neutral / sad
```

**Why this combination?**
- **wav2vec2** captures deep prosodic and phonetic patterns from 960h LibriSpeech pre-training
- **Statistical MFCC** provides interpretable acoustic features — energy, spectral shape, timbre
- **CREMA-D** adds 4,900 clips from 91 diverse actors, dramatically improving happy and sad generalisation
- **Smart augmentation** gives extra copies to neutral (imbalanced) and sad (most confused) classes

---

## 📁 Project Structure

```
speech-emotion-recognition/
├── config.py                 ← all settings
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── download_data.py      ← download RAVDESS + TESS + CREMA-D
│   ├── data_loader.py        ← parse all three datasets
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

### 4. Download datasets (RAVDESS + TESS + CREMA-D)
```bash
python src/download_data.py
```

### 5. Train
```bash
python src/train.py --force   # first run — extracts and caches features
python src/train.py           # subsequent runs — loads cache instantly
```

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

| Version | Approach | Dataset | Weighted F1 |
|---------|----------|---------|-------------|
| v1 | Deep CNN (2D MFCC maps) | RAVDESS | 0.13 |
| v2 | CNN + Data Augmentation | RAVDESS | 0.28 |
| v3 | Statistical MFCC + MLP | RAVDESS | 0.66 |
| v4 | Statistical MFCC + SVM | RAVDESS | 0.61 |
| v5 | SVM + TESS (wrong split) | R+T | 0.6557 |
| v6 | SVM + corrected split | R+T | 0.6579 |
| v7 | XGBoost + Pitch features | R+T | 0.61 |
| v8 | Rich MFCC + SVM + calibration | R+T | 0.6814 |
| v9 | wav2vec2 + MFCC + SVM | R+T | 0.7278 |
| **Final** | **wav2vec2 + MFCC + SVM + CREMA-D** | **R+T+C** | **0.7711 ✅** |

---

## 📦 Datasets

| Dataset | Speakers | Samples | Source |
|---------|----------|---------|--------|
| RAVDESS | 24 actors (12M, 12F) | 1,344 | [Zenodo](https://zenodo.org/record/1188976) |
| TESS | 2 female speakers | 2,000 | [Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) |
| CREMA-D | 91 actors (diverse) | 4,900 | [Kaggle](https://www.kaggle.com/datasets/ejlok1/cremad) |

**Evaluation protocol:** Speaker-independent — RAVDESS actors 1, 9, 12, 17, 19 held out exclusively for testing. All TESS and CREMA-D samples used for training only.

---

## 🔍 Critical Analysis

**Why all 4 emotions exceeded F1 = 0.72:**

Adding CREMA-D (91 actors, diverse demographics) provided the model with significantly more acoustic diversity for happy and sad — the two emotions that were previously confused (happy↔sad overlap in energy and pitch). The full RAVDESS dataset (speech + song, 1,344 samples) restored the proper 280-sample test set, giving statistically reliable per-class evaluation.

**Remaining challenges:**
1. Neutral has the fewest test samples (40 vs 80 for others) — precision is high (0.96) but recall lower (0.68)
2. Happy/sad still share some acoustic properties in acted speech corpora
3. Cross-dataset domain variation (studio vs acted vs crowd-sourced) requires diverse training data

---

## 📹 Video Presentation

**YouTube:** [YOUTUBE_LINK_HERE]

---

## 📄 Technical Report

Full IEEE-format technical report available in the repository:
`SER_Paper_Final_Mohammed_Natiq_Hilo.docx`

---

## 📚 References

1. A. Baevski et al., "wav2vec 2.0," NeurIPS, 2020.
2. S. R. Livingstone and F. A. Russo, "The RAVDESS dataset," PLOS ONE, 2018.
3. K. Dupuis and M. K. Pichora-Fuller, "TESS dataset," University of Toronto, 2010.
4. H. R. Cao et al., "CREMA-D dataset," IEEE Trans. Affective Computing, 2014.
5. B. Schuller et al., "The INTERSPEECH 2009 emotion challenge," Proc. Interspeech, 2009.

---

## 📜 License

Academic coursework — Al-Farabi University College, Department of Computer Engineering.
RAVDESS: CC BY-NC-SA 4.0. CREMA-D: Open Database License.
