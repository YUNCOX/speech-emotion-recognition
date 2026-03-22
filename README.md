# 🎙️ Speech Emotion Recognition
### Task #46 — Al-Farabi University | Department of Computer Engineering

> **Author:** Mohammed Natiq Hilo  
> **Approach:** wav2vec2 embeddings + Statistical MFCC + SVM  
> **Target:** Weighted F1-score > 0.72 on held-out speaker test set

---

## 🎯 Live Demo

**👉 [Try the Live Demo](https://b956579ca7bbf418fa.gradio.live)**

Record your voice and see the emotion predicted in real time with confidence scores and mel-spectrogram visualization.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Weighted F1** | **TBD after local training** |
| Accuracy | TBD |
| Macro F1 | TBD |

**Per-class F1:**

| Emotion | F1 | ≥ 0.72 |
|---------|----|--------|
| 😠 Angry   | TBD | — |
| 😊 Happy   | TBD | — |
| 😐 Neutral | TBD | — |
| 😢 Sad     | TBD | — |

---

## 🏗️ Architecture

```
Audio (.wav)
    ↓
wav2vec2-base (768-dim) ──┐
                           ├── Concatenate → ~1800-dim vector
Statistical MFCC (1043-dim)┘
    ↓
StandardScaler normalization
    ↓
Smart Augmentation (3-5x, neutral-weighted)
    ↓
SVM RBF (grid search C) + Neutral threshold calibration
    ↓
Emotion: angry / happy / neutral / sad
```

**Why this combination?**
- **wav2vec2** captures deep prosodic and phonetic patterns from 960h pre-training
- **Statistical MFCC** provides interpretable acoustic features (energy, pitch, timbre)
- **Neutral calibration** fixes the hardest class (only 40 test samples)
- **Smart augmentation** balances the neutral class imbalance

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

### 1. Install PyTorch with GPU support (RTX recommended)
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

### 5. Train (~5 min GPU / ~60 min CPU)
```bash
python src/train.py
```
Features are cached — re-runs are instant.

### 6. Run live demo
```bash
python app/demo.py --share    # public Gradio link
python app/demo.py            # local only
```

### 7. Predict single file
```bash
python src/predict.py --file path/to/audio.wav
```

---

## 🔬 Experimental Progression

| Version | Approach | Weighted F1 |
|---------|----------|-------------|
| v1 | Deep CNN (2D MFCC maps) | 0.13 |
| v2 | CNN + Augmentation | 0.28 |
| v3 | Statistical MFCC + MLP | 0.66 |
| v4 | Statistical MFCC + SVM | 0.61 |
| v5 | SVM + TESS (wrong split) | 0.48 |
| v6 | SVM + corrected split | 0.6579 |
| v7 | XGBoost + Pitch features | 0.61 |
| v8 | Rich MFCC + SVM + calibration | 0.6814 |
| **Final** | **wav2vec2 + MFCC + SVM + GPU** | **TBD** |

---

## 📦 Datasets

| Dataset | Speakers | Samples | Source |
|---------|----------|---------|--------|
| RAVDESS | 24 actors | 1,344 | [Zenodo](https://zenodo.org/record/1188976) |
| TESS | 2 female | ~2,800 | [Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) |

**Evaluation:** Speaker-independent — actors 1, 9, 12, 17, 19 held out for test only.

---

## 🔍 Critical Analysis

**Why F1 > 0.72 is challenging for all emotions:**

1. **Small speaker pool** — only 5 test speakers; one atypical speaker shifts metrics significantly
2. **Neutral class imbalance** — 40 neutral test samples vs 80 for others
3. **Cross-dataset domain mismatch** — TESS (studio clean) vs RAVDESS (acted speech)
4. **Speaker-independent evaluation** — most published results > 0.72 use speaker-dependent splits

---

## 📚 References

1. Baevski et al., "wav2vec 2.0," NeurIPS, 2020.
2. Livingstone & Russo, "RAVDESS dataset," PLOS ONE, 2018.
3. Dupuis & Pichora-Fuller, "TESS dataset," Univ. of Toronto, 2010.
4. Schuller et al., "INTERSPEECH 2009 emotion challenge," 2009.

---

## 📜 License

Academic coursework — Al-Farabi University. RAVDESS used under CC BY-NC-SA.
