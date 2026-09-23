# Human Activity Recognition using RNN, LSTM, and GRU

A deep learning project that compares the performance of **Simple RNN**, **LSTM**, and **GRU** architectures for **Human Activity Recognition (HAR)** using smartphone sensor data from the UCI HAR dataset.

## 📌 Overview

Human Activity Recognition (HAR) aims to identify human activities from sensor signals collected by mobile devices. This project evaluates and compares three recurrent neural network architectures:

- Simple RNN
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

The models are trained on multivariate time-series sensor data and evaluated using accuracy, precision, recall, F1-score, confusion matrices, and computational efficiency.

---

## 📂 Dataset

**Dataset:** UCI Human Activity Recognition (HAR)

The dataset contains recordings from smartphone accelerometer and gyroscope sensors.

### Activities

- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

### Input Shape

```python
(128, 9)
```

- 128 time steps
- 9 sensor channels

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 🏗️ Model Architectures

### 1️⃣ Simple RNN

```python
SimpleRNN(32)
Dropout(0.2)
Dense(6, activation='softmax')
```

### 2️⃣ LSTM

```python
LSTM(32)
Dropout(0.2)
Dense(6, activation='softmax')
```

### 3️⃣ GRU

```python
GRU(32)
Dropout(0.2)
Dense(6, activation='softmax')
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|------------|--------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Sparse Categorical Crossentropy |
| Epochs | 30 |
| Batch Size | 32 |
| Random Seed | 42 |

---

## 📊 Evaluation Metrics

The following metrics are used to evaluate each model:

- Accuracy
- Precision
- Recall
- F1 Score
- Training Time
- Number of Parameters
- Confusion Matrix

---

## 📈 Results

The project analyzes:

- Training vs Validation Loss
- Training vs Validation Accuracy
- Confusion Matrices
- Model Comparison
- Computational Cost
- Sequence Length Sensitivity

### Key Observations

- LSTM and GRU outperform Simple RNN on most activities.
- LAYING is consistently the easiest activity to recognize.
- SITTING and STANDING are the most frequently confused activities.
- GRU provides competitive accuracy with lower computational complexity compared to LSTM.
- All models show good generalization with minimal overfitting.

---

## 📋 Project Workflow

```text
Load Dataset
      ↓
Preprocess Data
      ↓
Train RNN
      ↓
Train LSTM
      ↓
Train GRU
      ↓
Evaluate Models
      ↓
Generate Metrics & Visualizations
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/HAR-RNN-LSTM-GRU.git
cd HAR-RNN-LSTM-GRU
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook RNN_LSTM_GRU.ipynb
```

or

```bash
jupyter lab
```

Open the notebook and run all cells.

---



## 🎯 Learning Outcomes

- Understanding sequence modeling using recurrent neural networks.
- Comparing RNN, LSTM, and GRU architectures.
- Analyzing confusion matrices and classification errors.
- Evaluating model convergence and computational efficiency.
- Applying deep learning to time-series sensor data.

---
