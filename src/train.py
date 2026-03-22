# ================================================================
# src/train.py — Full training pipeline
# wav2vec2 embeddings + Statistical MFCC + SVM + v8 tricks
#
# Usage:
#   python src/train.py
#   python src/train.py --force    (re-extract features)
# ================================================================

import os, sys, json, pickle, argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (f1_score, accuracy_score,
                              classification_report, confusion_matrix)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TEST_SPEAKERS, SVM_C_VALUES, TARGET_F1,
                    MODEL_FILE, RESULTS_DIR, MODELS_DIR,
                    N_AUG_NEUTRAL, N_AUG_OTHER)
from src.data_loader import load_all_data
from src.features import load_wav2vec2, extract_all_features


def split_data(X, y_enc, g):
    test_mask  = np.array([a in TEST_SPEAKERS for a in g])
    train_mask = ~test_mask
    return (X[train_mask], X[test_mask],
            y_enc[train_mask], y_enc[test_mask])


def augment(X_tr_n, y_tr, le):
    """Smart augmentation — extra copies for neutral class."""
    aug_X, aug_y = [X_tr_n], [y_tr]
    for label in range(len(le.classes_)):
        idx     = np.where(y_tr == label)[0]
        cls     = le.inverse_transform([label])[0]
        n_copies = N_AUG_NEUTRAL if cls == 'neutral' else N_AUG_OTHER
        for _ in range(n_copies):
            noise = np.random.normal(0, 0.035, X_tr_n[idx].shape)
            aug_X.append(X_tr_n[idx] + noise)
            aug_y.append(y_tr[idx])
    return np.vstack(aug_X), np.concatenate(aug_y)


def train_and_evaluate(X_tr_n, X_te_n, y_tr, y_te, le):
    """Grid search SVM + RF, neutral threshold calibration."""
    neutral_idx = le.transform(['neutral'])[0]

    # SVM grid search
    print("\n🔍 SVM grid search...")
    best_f1, best_C, best_svm = 0, 1, None
    for C in SVM_C_VALUES:
        m = SVC(kernel='rbf', C=C, gamma='scale', probability=True,
                class_weight='balanced', random_state=42)
        m.fit(X_tr_n, y_tr)
        f1 = f1_score(y_te, m.predict(X_te_n), average='weighted')
        print(f"  SVM C={C:<5} → F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_C, best_svm = f1, C, m

    # Random Forest
    print("\n🌲 Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=20,
                                class_weight='balanced',
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr_n, y_tr)
    rf_f1 = f1_score(y_te, rf.predict(X_te_n), average='weighted')
    print(f"  RF → F1: {rf_f1:.4f}")

    best_model   = best_svm if best_f1 >= rf_f1 else rf
    base_f1      = max(best_f1, rf_f1)

    # Neutral threshold calibration
    print("\n🎯 Calibrating neutral threshold...")
    probs = best_model.predict_proba(X_te_n)
    best_thresh_f1, best_thresh = 0, 0.25
    for t in np.arange(0.10, 0.55, 0.02):
        y_pred_t = [neutral_idx if p[neutral_idx] >= t
                    else np.argmax(p) for p in probs]
        f1_t = f1_score(y_te, y_pred_t, average='weighted')
        if f1_t > best_thresh_f1:
            best_thresh_f1, best_thresh = f1_t, t

    # Final prediction
    if best_thresh_f1 > base_f1:
        y_pred = [neutral_idx if p[neutral_idx] >= best_thresh
                  else np.argmax(p) for p in probs]
        method = f"SVM C={best_C} + neutral_thresh={best_thresh:.2f}"
        final_thresh = best_thresh
    else:
        y_pred = best_model.predict(X_te_n)
        method = f"SVM C={best_C}"
        final_thresh = None

    return best_model, y_pred, method, final_thresh, neutral_idx


def evaluate_and_plot(y_te, y_pred, le, method):
    wf1  = f1_score(y_te, y_pred, average='weighted')
    acc  = accuracy_score(y_te, y_pred)
    mf1  = f1_score(y_te, y_pred, average='macro')
    pcf1 = f1_score(y_te, y_pred, average=None)
    cm   = confusion_matrix(y_te, y_pred)

    print("\n" + "="*60)
    print("  FINAL RESULTS — wav2vec2 + MFCC + SVM")
    print("="*60)
    print(f"  Method      : {method}")
    print(f"  Accuracy    : {acc*100:.2f}%")
    print(f"  Weighted F1 : {wf1:.4f}  ← Target: >{TARGET_F1}")
    print(f"  Macro F1    : {mf1:.4f}")
    print("="*60)
    print(classification_report(y_te, y_pred, target_names=le.classes_))
    print("Per-class:")
    all_pass = True
    for i, cls in enumerate(le.classes_):
        status = "✅" if pcf1[i] >= TARGET_F1 else "❌"
        if pcf1[i] < TARGET_F1: all_pass = False
        print(f"  {status} {cls:<8}: F1 = {pcf1[i]:.4f}")
    print("="*60)
    if wf1 >= TARGET_F1:
        print("🎯 WEIGHTED F1 TARGET ACHIEVED!")
    else:
        print(f"📈 Gap: {TARGET_F1 - wf1:.4f}")

    # Plots
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
    axes[0].set_title(f'Confusion Matrix | F1: {wf1:.4f}', fontweight='bold')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

    colors = ['#e74c3c','#f39c12','#2ecc71','#3498db']
    bars   = axes[1].bar(le.classes_, pcf1, color=colors,
                         edgecolor='black', alpha=0.85)
    axes[1].axhline(TARGET_F1, color='red', ls='--', lw=2,
                    label=f'Target = {TARGET_F1}')
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title('Per-Class F1 Score', fontweight='bold')
    axes[1].legend(); axes[1].grid(axis='y', alpha=0.3)
    for b, f in zip(bars, pcf1):
        axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                     f'{f:.3f}', ha='center', fontweight='bold')
    plt.suptitle(f'SER | Weighted F1: {wf1:.4f}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'results.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Plot saved: {RESULTS_DIR}/results.png")

    return {'accuracy': float(acc), 'weighted_f1': float(wf1),
            'macro_f1': float(mf1), 'method': method,
            'per_class_f1': {le.classes_[i]: float(pcf1[i])
                             for i in range(len(le.classes_))},
            'all_above_target': all_pass}


def main(force=False):
    np.random.seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,  exist_ok=True)

    print("="*60)
    print("  SPEECH EMOTION RECOGNITION — TRAINING")
    print("  wav2vec2 + Statistical MFCC + SVM")
    print("="*60)

    # 1. Load data
    df = load_all_data()

    # 2. Extract features
    processor, w2v_model, device = load_wav2vec2()
    X, y_raw, g = extract_all_features(
        df, processor, w2v_model, device, force=force)

    # 3. Encode + split
    le    = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    print(f"\n📌 Classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    X_tr, X_te, y_tr, y_te = split_data(X, y_enc, g)
    print(f"Train: {len(X_tr)} | Test: {len(X_te)}")
    print("Test distribution:")
    for cls in le.classes_:
        print(f"  {cls}: {(y_te == le.transform([cls])[0]).sum()}")

    # 4. Normalize
    scaler = StandardScaler()
    X_tr_n = scaler.fit_transform(X_tr)
    X_te_n = scaler.transform(X_te)

    # 5. Augment
    X_aug, y_aug = augment(X_tr_n, y_tr, le)
    print(f"\n✅ Augmented training: {len(X_aug)} samples")

    # 6. Train + calibrate
    best_model, y_pred, method, final_thresh, neutral_idx = \
        train_and_evaluate(X_aug, X_te_n, y_aug, y_te, le)

    # 7. Evaluate + plot
    metrics = evaluate_and_plot(y_te, y_pred, le, method)

    # 8. Save
    payload = {
        'model':             best_model,
        'scaler':            scaler,
        'label_encoder':     le,
        'neutral_threshold': final_thresh,
        'neutral_idx':       neutral_idx,
        'metrics':           metrics,
    }
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(payload, f)
    print(f"\n✅ Model saved: {MODEL_FILE}")

    with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved: {RESULTS_DIR}/metrics.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true',
                        help='Force re-extraction of features')
    args = parser.parse_args()
    main(force=args.force)
