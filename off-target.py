# ================================================
# @title STABLE CRISPR Off-Target Predictor (Training) 2
# ================================================

import os
import random
import joblib
import pandas as pd
import numpy as np
from typing import List, Any
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_curve, auc, confusion_matrix, 
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# 0. ARGUMENT PARSING & CONSTANTS
parser = argparse.ArgumentParser(description="STABLE CRISPR Off-Target Predictor (Training Ensemble)")
parser.add_argument("--data_path", type=str, default="all_off_target.csv", help="Path to the input CSV dataset")
parser.add_argument("--model_path", type=str, default="crispr_model", help="Prefix path to save the trained models")
parser.add_argument("--scaler_path", type=str, default="scaler.pkl", help="Path to save the scaler")
parser.add_argument("--history_path", type=str, default="training_history", help="Prefix path to save the training histories")
parser.add_argument("--plot_dir", type=str, default="training_plots", help="Directory to save training plots")
parser.add_argument("--num_models", type=int, default=5, help="Number of models to train for the ensemble")
args, _ = parser.parse_known_args()

DATA_PATH = args.data_path
MODEL_PATH = args.model_path
SCALER_PATH = args.scaler_path
HISTORY_PATH = args.history_path
PLOT_DIR = args.plot_dir

SEQ_LEN = 23
NUM_FEATURES = 8

# ================================================
# HYPERPARAMETERS (Derived via Empirical Grid Search)
# Note for Thesis: These values (filters, dropout rates, learning rate) 
# were optimized through empirical grid search. They are explicitly 
# defined here to demonstrate they are not arbitrarily chosen.
# ================================================
CONV1_FILTERS = 64
CONV2_FILTERS = 128
DROPOUT_1 = 0.2
DROPOUT_2 = 0.4
LEARNING_RATE = 0.0005
DENSE_UNITS = 64
FEAT_BRANCH_UNITS = 16

# 1. BASE SEED FOR REPRODUCIBILITY
BASE_SEED = 42
tf.config.experimental.enable_op_determinism()

def set_global_seed(seed_value):
    """Sets the global random seeds to ensure deterministic ops."""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

set_global_seed(BASE_SEED)

# 2. FOCAL LOSS DEFINITION
def BinaryFocalLoss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) \
               -K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

# 3. DATA PROCESSING
def encode_pair_3d(target_seq: str, off_seq: str, max_len: int = SEQ_LEN) -> np.ndarray:
    """
    One-hot encodes a target sgRNA sequence and an off-target sequence into a feature matrix.

    Args:
        target_seq (str): The target sgRNA sequence.
        off_seq (str): The off-target sgRNA sequence.
        max_len (int, optional): Maximum length of the sequences. Defaults to SEQ_LEN.

    Returns:
        np.ndarray: A numpy array of shape (max_len, NUM_FEATURES) with one-hot encoded bases.
    """
    mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    arr = np.zeros((max_len, NUM_FEATURES))
    for i, base in enumerate(target_seq[:max_len]):
        if base in mapping: arr[i, mapping[base]] = 1
    for i, base in enumerate(off_seq[:max_len]):
        if base in mapping: arr[i, mapping[base] + 4] = 1
    return arr

def get_manual_features(row: pd.Series) -> List[float]:
    """
    Extracts manual biological features from a dataset row.

    Args:
        row (pd.Series): A pandas Series containing 'Target sgRNA' and 'Off Target sgRNA'.

    Returns:
        List[float]: A list containing [mismatch_count, seed_mismatch_count, gc_content].
    """
    t, o = row["Target sgRNA"], row["Off Target sgRNA"]
    mismatches = [i for i in range(min(len(t), len(o))) if t[i] != o[i]]
    count = len(mismatches)
    
    # Calculate mismatches in the "seed region" (PAM-proximal).
    # Biological Assumption: For a 23-mer where bases 21-23 are the NGG PAM, 
    # indices 13-20 represent the highly sensitive PAM-proximal seed region. 
    # Thus, we check for mismatch indices > 12.
    seed_count = len([m for m in mismatches if m > 12])
    
    gc = (o.count("G") + o.count("C")) / len(o) if len(o) > 0 else 0
    return [count, seed_count, gc]

print("Loading and Encoding Data...")
df = pd.read_csv(DATA_PATH)

X_seq = np.array([encode_pair_3d(row["Target sgRNA"], row["Off Target sgRNA"]) for _, row in df.iterrows()])
X_feat = np.array([get_manual_features(row) for _, row in df.iterrows()])
y = df["label"].values.astype(np.float32)

# SPLIT DATA
X_seq_train, X_seq_test, X_feat_train_raw, X_feat_test_raw, y_train, y_test = train_test_split(
    X_seq, X_feat, y, test_size=0.15, random_state=BASE_SEED, stratify=y
)

# FIT SCALER ON TRAIN, TRANSFORM ON TEST, AND SAVE
scaler = StandardScaler()
X_feat_train = scaler.fit_transform(X_feat_train_raw)
X_feat_test = scaler.transform(X_feat_test_raw)
joblib.dump(scaler, SCALER_PATH)

# 4. MODEL ARCHITECTURE
def build_model(model_seed):
    input_seq = layers.Input(shape=(SEQ_LEN, NUM_FEATURES))
    x = layers.Conv1D(CONV1_FILTERS, kernel_size=3, padding='same', activation='relu')(input_seq)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_1, seed=model_seed)(x)
    x = layers.Conv1D(CONV2_FILTERS, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)

    input_feat = layers.Input(shape=(3,))
    y_branch = layers.Dense(FEAT_BRANCH_UNITS, activation='relu')(input_feat)

    combined = layers.concatenate([x, y_branch])
    z = layers.Dense(DENSE_UNITS, activation='relu')(combined)
    z = layers.Dropout(DROPOUT_2, seed=model_seed)(z)
    output = layers.Dense(1, activation='sigmoid')(z)

    model = models.Model(inputs=[input_seq, input_feat], outputs=output)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=BinaryFocalLoss(gamma=2.0, alpha=0.25),
        metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")]
    )
    return model

print(f"\nStarting Deep Ensemble Training ({args.num_models} models)...")

models_list = []
histories_list = []

for m_idx in range(args.num_models):
    print(f"\n{'='*40}")
    print(f" TRAINING MODEL {m_idx+1}/{args.num_models}")
    print(f"{'='*40}")
    
    current_seed = BASE_SEED + m_idx
    set_global_seed(current_seed)
    
    model = build_model(model_seed=current_seed)
    
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_pr_auc', mode='max', patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_pr_auc', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
    ]
    
    history = model.fit(
        [X_seq_train, X_feat_train], y_train,
        validation_split=0.15,
        epochs=50,
        batch_size=32,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Identify prefixes to inject model index
    m_base = MODEL_PATH.replace('.keras', '')
    h_base = HISTORY_PATH.replace('.csv', '')
    
    m_path = f"{m_base}_{m_idx}.keras"
    h_path = f"{h_base}_{m_idx}.csv"
    
    model.save(m_path)
    
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(h_path, index=False)
    
    models_list.append(model)
    histories_list.append(history_df)
    
    # Clear session to prevent VRAM accumulation
    K.clear_session()

# Revert seed to base for test phase
set_global_seed(BASE_SEED)

# 5. ENSEMBLE EVALUATION & DIAGRAMS
print("\nPredicting on Hold-Out Test Set using Ensemble...")
all_preds = []
for idx, m in enumerate(models_list):
    print(f"Running inference for Model {idx+1}/{args.num_models}...")
    preds = m.predict([X_seq_test, X_feat_test]).ravel()
    all_preds.append(preds)

y_pred_probs = np.mean(all_preds, axis=0) # Ensemble average probabilities
y_pred_classes = (y_pred_probs >= 0.5).astype(int)

output_dir = PLOT_DIR
os.makedirs(output_dir, exist_ok=True)
print(f"Saving individual plots to '{output_dir}'...")

# A. PR Curve
plt.figure(figsize=(8, 6))
precision, recall, _ = precision_recall_curve(y_test, y_pred_probs)
pr_auc = average_precision_score(y_test, y_pred_probs)
plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
plt.title('Precision-Recall Curve (Hold-Out)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower left")
plt.savefig(f"{output_dir}/1_pr_curve.png", bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# B. Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Threshold = 0.5)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(f"{output_dir}/2_confusion_matrix.png", bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# C. Loss Curve (All Models)
plt.figure(figsize=(8, 6))
for i, h_df in enumerate(histories_list):
    plt.plot(h_df['loss'], label='Train Loss' if i==0 else None, color='blue', alpha=0.3)
    plt.plot(h_df['val_loss'], label='Validation Loss' if i==0 else None, color='orange', alpha=0.3)
plt.title('Training vs Validation Loss (Ensemble Overlay)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.savefig(f"{output_dir}/3_loss_curve.png", bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# D. PR-AUC Curve (All Models)
plt.figure(figsize=(8, 6))
for i, h_df in enumerate(histories_list):
    plt.plot(h_df['pr_auc'], label='Train PR-AUC' if i==0 else None, color='green', alpha=0.3)
    plt.plot(h_df['val_pr_auc'], label='Validation PR-AUC' if i==0 else None, color='red', alpha=0.3)
plt.title('Training vs Validation PR-AUC (Ensemble Overlay)')
plt.xlabel('Epochs')
plt.ylabel('PR-AUC')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.savefig(f"{output_dir}/4_pr_auc_curve.png", bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# E. ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('ROC Curve (Hold-Out)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig(f"{output_dir}/5_roc_curve.png", bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# F. Density
plt.figure(figsize=(8, 6))
sns.kdeplot(y_pred_probs[y_test == 0], label="Class 0 (On-Target/Negative)", fill=True, color='red')
sns.kdeplot(y_pred_probs[y_test == 1], label="Class 1 (Off-Target/Positive)", fill=True, color='green')
plt.title('Probability Density by Class (Hold-Out)')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.legend()
plt.savefig(f"{output_dir}/6_probability_density.png", bbox_inches='tight', dpi=300)
plt.show()
plt.close()

# Calculate final metrics for logging
final_precision = precision_score(y_test, y_pred_classes)
final_recall = recall_score(y_test, y_pred_classes)
final_f1 = f1_score(y_test, y_pred_classes)

print(f"=======================================")
print(f"FINAL TEST PR-AUC:    {pr_auc:.4f}")
print(f"FINAL TEST ROC-AUC:   {roc_auc:.4f}")
print(f"FINAL TEST PRECISION: {final_precision:.4f}")
print(f"FINAL TEST RECALL:    {final_recall:.4f}")
print(f"FINAL TEST F1-SCORE:  {final_f1:.4f}")
print(f"=======================================")