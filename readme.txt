Speech Emotion Recognition (SER) on RAVDESS
==========================================

Overview
--------
This project builds a speech emotion recognition system on the RAVDESS dataset.
Pipeline: actor-wise split (to avoid speaker leakage) → audio to log-mel spectrograms →
z-score normalization (fit on train only) → CNN+Transformer (ParallelModel) →
train with AdamW+ReduceLROnPlateau and light SpecAugment → evaluate on a held-out test set.

Dataset & Split
---------------
- Source: RAVDESS (speech subset; 48 kHz WAV).
- Each clip is trimmed/padded to 3.0 s for fixed input length.
- Split is *speaker-independent*: 80% actors for train, 10% for val, 10% for test.
  (No actor overlap across splits — prevents leakage and optimistic metrics.)

Features & Preprocessing
------------------------
- Features: 128-bin log-mel spectrograms (n_fft=1024, hop=256).
- Standardization: flatten (B, 1*128*T), fit StandardScaler on *train only*;
  transform val/test and inference with the same scaler (saved alongside the model).
- Optional augmentation on train only: SpecAugment (time & freq masks) and AWGN.

Model
-----
- ParallelModel: 2D CNN stack (Conv+BN+ReLU+MaxPool+Dropout) in parallel with a
  TransformerEncoder (d_model=64). The two embeddings are concatenated and fed
  to a Linear classifier (implemented as LazyLinear to adapt to input time length).
- Input shape: (B, 1, 128, T) after mel; T depends on clip length/hop.
- Loss: CrossEntropy with class weights + mild label smoothing (0.05).

Training
--------
- Optimizer: AdamW (lr=3e-4, weight_decay=1e-4).
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5) on validation loss.
- Epochs: 150–300 with early stopping (patience≈10–15).
- Batch size: 32 (adjust by GPU/CPU memory).

Evaluation 
-----------------------------------
              precision    recall  f1-score   support

   surprised      0.833     0.897     0.864        39
     neutral      0.615     0.800     0.696        20
        calm      0.905     0.974     0.938        39
       happy      0.969     0.795     0.873        39
         sad      0.968     0.769     0.857        39
       angry      0.949     0.949     0.949        39
     fearful      0.897     0.897     0.897        39
     disgust      0.833     0.897     0.864        39

    accuracy                          0.877       293
   macro avg      0.871     0.872     0.867       293
weighted avg      0.888     0.877     0.878       293



How to Reproduce
----------------
1 Prepare RAVDESS speech WAVs.
2 Run notebook cells in order:
   - Build metadata (paths, labels) and perform *actor-wise* split.
   - Precompute log-mel features and apply StandardScaler (fit on train only).
   - Build DataLoaders on the normalized features.
   - Define ParallelModel (use LazyLinear as the final classifier layer).
   - Train with AdamW + scheduler + SpecAugment (train only).
   - Save: model weights (best_parallel_ser.pt) and scaler (mel_scaler.pkl).
3 Inference: use the same scaler to transform the mel of a single WAV
   (trim/pad to 3s, resample to 48 kHz) and call model to get class probabilities.

Known Limitations (Real-World Gap)
----------------------------------
- RAVDESS is *acted* emotion; real-world spontaneous speech is noisier and more nuanced.
- Domain shift: microphones, sampling rates, room acoustics, languages/dialects
  differ from the training data → performance drops without adaptation.
- Class imbalance: some emotions have fewer samples and are harder to learn.
- Single-label framing: people often express mixed emotions; multi-label or
  continuous (valence/arousal) formulations may be more appropriate.

Potential Improvements
----------------------
- Stronger augmentation: room impulse responses, real-world noise, pitch/tempo jitter.
- Speaker normalization: per-speaker CMVN or adversarial speaker-invariant training.
- Better features: log-mel + deltas, or learnable front-ends.
- Model tweaks: SpecAugment tuning, longer context, mixup/CutMix on spectrograms.
- Data: add other corpora (e.g., CREMA-D, TESS) and perform domain adaptation.
- Calibration: temperature scaling for better probability estimates.

References
----------
- RAVDESS dataset.
- Prior SER baselines and public tutorials 
[1] D. Kosta, "Speech-Emotion-Classification-with-PyTorch," 2021. [Online]. Available: https://github.com/Data-Science-kosta/Speech-Emotion-Classification-with-PyTorch
