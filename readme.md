# 🧠 ML-09 — Machine Learning Project Collection

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Classic%20ML-EB5E28?logo=xgboost&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-completed-brightgreen)

> **ML9** is a two-project collection exploring machine learning across two very different domains: **financial time-series forecasting** (Bitcoin price prediction) and **computer vision** (image classification on Caltech-101).

---

## 📑 Table of Contents

- [📦 Part 1 — Bitcoin Price Prediction](#-part-1--bitcoin-price-prediction)
  - [1.1 `classic_models.ipynb` — XGBoost Baseline](#11-classic_modelsipynb--xgboost-baseline)
  - [1.2 `new_models.ipynb` — Neural Network vs. XGBoost](#12-new_modelsipynb--neural-network-vs-xgboost)
  - [1.3 Comparison: Classic Models vs. New Models](#13-comparison-classic-models-vs-new-models)
- [🖼️ Part 2 — Image Classification on Caltech-101](#️-part-2--image-classification-on-caltech-101)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 How to Run](#-how-to-run)

---

## 📦 Part 1 — Bitcoin Price Prediction

This part contains two notebooks tackling the **same problem** — predicting Bitcoin's next-day price direction and next-day closing price from daily OHLCV data — using two different modeling philosophies, so their results can be directly compared.

### 1.1 `classic_models.ipynb` — XGBoost Baseline

**What it does:**
This notebook builds the *classical machine learning* baseline for the Bitcoin forecasting problem.

**Steps:**
1. Loads `bitcoin_dataset.csv` — 1,461 daily rows (2020–2024) with `Open, High, Low, Close, Volume`.
2. Engineers features:
   - `benefit = Close - Open`
   - `class` = 1 if that day closed up, else 0
   - `y` = next day's `class` (classification target)
   - `y_reg` = next day's `Close` price (regression target)
3. Splits data chronologically (80% train / 20% test, no shuffling — important for time series).
4. Trains two separate models:
   - **`XGBClassifier`** → predicts next-day direction (up/down)
   - **`XGBRegressor`** → predicts next-day closing price

**Tools & libraries used:** `pandas`, `scikit-learn` (`train_test_split`, metrics), `xgboost`

**Model used:** XGBoost (Extreme Gradient Boosting) — a tree-based ensemble model, for both classification and regression tasks.

**Results:**

| Task | Metric | Score |
|------|--------|-------|
| Classification | Accuracy | **52.74%** |
| Classification | F1-score (macro avg) | 0.51 |
| Regression | RMSE | **$1,675.55** |
| Training time (classification) | — | 0.53s |
| Training time (regression) | — | 0.11s |

---

### 1.2 `new_models.ipynb` — Neural Network vs. XGBoost

**What it does:**
This notebook repeats the exact same Bitcoin forecasting task, but introduces a **neural network** and directly benchmarks it against XGBoost inside the same notebook.

**Steps:**
1. Rebuilds the same dataset and features as `classic_models.ipynb` (`benefit`, `class`, `y`, `y_reg`).
2. Scales features with `StandardScaler` (required for neural networks to train well) — both `X` and the regression target `y_reg` are scaled.
3. Builds and trains two **Keras Sequential MLPs** (Multi-Layer Perceptrons):
   - Classification MLP: `Dense(16, relu) → Dense(8, relu) → Dense(1, sigmoid)`, trained 30 epochs
   - Regression MLP: same hidden structure, linear output, trained 30 epochs on scaled targets (inverse-transformed back to dollars afterward)
4. Re-trains XGBoost **inside the same notebook** on the identical split, so both models are benchmarked fairly side by side.
5. Builds a final `comparison_df` table printing time and metrics for both models.

**Tools & libraries used:** `pandas`, `scikit-learn` (`StandardScaler`, metrics), `tensorflow.keras` (`Sequential`, `Dense`, `Input`), `xgboost`

**Models used:** MLP Neural Network (Keras/TensorFlow) **and** XGBoost (for direct comparison).

**Results:**

| Task | Model | Time (s) | Metric |
|------|-------|----------|--------|
| Classification | XGBoost | **0.37s** | Accuracy: **52.74%** |
| Classification | MLP Neural Net | 3.69s | Accuracy: 49.66% |
| Regression | XGBoost | **0.08s** | RMSE: $1,675.55 |
| Regression | MLP Neural Net | 3.25s | RMSE: **$1,367.23** |

---

### 1.3 Comparison: Classic Models vs. New Models

| Aspect | `classic_models.ipynb` | `new_models.ipynb` |
|--------|------------------------|---------------------|
| Purpose | Establish XGBoost baseline | Compare XGBoost vs. Neural Net head-to-head |
| Preprocessing | Raw features, no scaling | Features & targets scaled with `StandardScaler` |
| Models | XGBoost only | XGBoost **+** Keras MLP |
| Classification accuracy | 52.74% | XGBoost: 52.74% · MLP: 49.66% |
| Regression RMSE | $1,675.55 | XGBoost: $1,675.55 · MLP: $1,367.23 |
| Training speed | Fast (sub-second) | XGBoost fast · MLP ~10× slower |

**Conclusion:**
- 🌳 For **classification**, XGBoost matched the neural network's job and even slightly outperformed it — while training **~10× faster**. On this small, tabular dataset, the extra complexity of a neural net didn't pay off.
- 🔁 For **regression**, the neural network actually achieved a **lower RMSE** ($1,367 vs. $1,675), suggesting it captured some non-linear relationship in price magnitude that the tree-based model missed.
- ⚖️ Overall, neither model is a universal winner — the right choice depends on the specific sub-task (classification vs. regression), reinforcing that **more complex ≠ always better**.

---

## 🖼️ Part 2 — Image Classification on Caltech-101

**Notebook:** `caltech_101.ipynb`

**What it does:**
This notebook trains a **custom Convolutional Neural Network (CNN) from scratch** to classify images from the **Caltech-101** dataset — a well-known benchmark containing images across 102 object categories (airplanes, faces, chairs, animals, etc.).

**Steps:**
1. **Setup:** Detects and uses GPU acceleration if available (ran on an NVIDIA GTX 1650 Ti in this case).
2. **Data loading:** Loads the dataset via `torchvision.datasets.ImageFolder` — **102 classes, 9,144 total images**.
3. **Preprocessing & augmentation:**
   - Resize all images to 128×128
   - Training set: random horizontal flip + random rotation (±15°) for data augmentation
   - Normalization to [-1, 1] range
4. **Split:** 80% training / 20% validation.
5. **Model architecture — `StudentCNN`:**
   - 3 convolutional blocks: `Conv2d → ReLU → MaxPool`, channels growing 32 → 64 → 128
   - Flatten → `Dense(256)` → Dropout(0.5) → `Dense(num_classes)` output layer
6. **Training:** 10 epochs, Adam optimizer (lr=0.001, weight decay for regularization), cross-entropy loss.
7. **Evaluation & visualization:** Plots training/validation loss and accuracy curves across epochs to show convergence behavior.

**Tools & libraries used:** `torch`, `torchvision`, `PIL` (image loading), `matplotlib` (plotting)

**Model used:** Custom-built CNN (`StudentCNN`) — trained entirely from scratch, no pretrained weights.

**Results:**

| Epoch | Train Accuracy | Val Accuracy |
|-------|-----------------|---------------|
| 1 | 23.57% | 36.30% |
| 4 | 46.18% | 53.03% |
| 10 (final) | — | **59.87%** |

**What this shows:**
Reaching ~60% validation accuracy across **102 fine-grained classes**, using a simple CNN with no pretrained backbone, is a solid result — it demonstrates that the model is genuinely learning meaningful visual features (given that random guessing would be ~1%). The steadily narrowing gap between train and validation curves also indicates the augmentation strategy (flip + rotation) is helping generalization rather than overfitting.

---

## 🛠️ Tech Stack

- **Data & Classical ML:** `pandas`, `numpy`, `scikit-learn`, `xgboost`
- **Deep Learning:** `tensorflow` / `keras`, `torch`, `torchvision`
- **Visualization:** `matplotlib`
- **Environment:** Jupyter Notebook, Python 3

---

## 🚀 How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost tensorflow torch torchvision matplotlib pillow

# Launch notebooks
jupyter notebook
```

Then open any notebook and run the cells top to bottom:
- `classic_models.ipynb` / `new_models.ipynb` require `bitcoin_dataset.csv` in the working directory.
- `caltech_101.ipynb` requires the Caltech-101 image folder (`./caltech-101` or `./101_ObjectCategories`) in the working directory.

---

<p align="center">Made with 🐍 Python, 🌳 XGBoost, 🔥 PyTorch, and a lot of ☕</p>