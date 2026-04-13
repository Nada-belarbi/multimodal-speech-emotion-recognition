# Multimodal Speech Emotion Recognition

This project investigates speech emotion recognition (SER) under realistic evaluation settings, with a focus on multimodal learning and generalization across speakers.

---

## 🎯 Objective

The goal of this project is to better understand:

- whether models truly learn emotional patterns or speaker-specific features
- how evaluation protocols impact performance
- whether combining audio and text improves emotion recognition

---

## 📊 Approach

We explore three configurations:

- Audio-only models (acoustic features)
- Text-only models (TF-IDF and dense embeddings)
- Multimodal models (audio + text fusion)

Evaluation is performed using:

- GroupKFold (speaker-independent)
- Leave-One-Speaker-Out (LOSO)

---
## Multimodal Fusion Strategy

To combine audio and textual information, several fusion strategies were explored:

- **Early Fusion**:  
  Audio and text features are concatenated into a single representation before being fed into the model. This provides a simple baseline for multimodal learning.

- **Adaptive Weighted Fusion**:  
  Modality-specific contributions are weighted to balance the influence of audio and text features.

- **Cross-Modal Interaction (Extended Model)**:  
  Additional interaction terms are introduced to capture relationships between audio and textual representations.

These approaches allow us to evaluate how different levels of interaction between modalities impact model performance and generalization.
---

## 🧠 Key Findings

- Audio-only models provide the strongest and most robust performance
- Text-only models perform significantly worse
- Multimodal fusion does not consistently improve performance

💡 Insight:
Multimodal learning is not inherently superior and depends heavily on data and fusion strategy.

---

## 🔬 Relation to Previous Work

This project extends a previous study on RAVDESS, where a strong performance drop was observed under speaker-independent evaluation.

In contrast, IEMOCAP shows:

- smaller performance drops
- higher variability
- more realistic conditions

This highlights the importance of dataset characteristics in evaluating model generalization.

---

## 🛠️ Methods

- Audio features: MFCC-based representations
- Text features: TF-IDF + dense embeddings
- Models:
  - Random Forest
  - MLP
  - CNN (audio baseline)

---

## 🚀 Future Work

- Advanced multimodal fusion (attention, transformers)
- Sequential modeling (LSTM, temporal models)
- Cross-dataset evaluation
- Fine-tuning text embeddings for emotion recognition

---

## 📌 Keywords

Speech Emotion Recognition, Multimodal Learning, Audio Processing, NLP, Deep Learning, Robust AI

---

## 👤 Author

Nada Belarbi  
AI & Machine Learning Engineer

---

## 🔗 Connect

- LinkedIn: [https://www.linkedin.com/in/nada-belarbi-431884272/]
- GitHub: [https://github.com/Nada-belarbi/]
