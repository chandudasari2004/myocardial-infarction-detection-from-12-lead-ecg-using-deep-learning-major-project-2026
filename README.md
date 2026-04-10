# 🫀 Myocardial Infarction Detection from 12-Lead ECG using Deep Learning

## 📌 Overview

This project presents a **deep learning-based system** for detecting **Myocardial Infarction (MI)** (heart attack) using **12-lead ECG signals**.

The system uses a **hybrid model combining 1D Convolutional Neural Network (CNN) and Bidirectional Long Short-Term Memory (BiLSTM)** to automatically learn:

* Spatial features (ECG waveform patterns)
* Temporal features (heartbeat sequence patterns)

The model classifies ECG signals into:

* ✅ **MI (Myocardial Infarction)**
* ✅ **Non-MI (Healthy)**

---

## 📊 Dataset

* **Dataset Used:** PTB Diagnostic ECG Database (PhysioNet)
* **Total Patients:** 294
* **Leads:** 12-lead ECG
* **Sampling Rate:** 1000 Hz

🔗 Dataset Download (used in code):

```bash
!wget -r -N -c -np https://physionet.org/files/ptbdb/1.0.0/
```

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing

* ECG signals loaded using `wfdb`
* Normalization:

  * Mean subtraction
  * Standard deviation scaling
* Segmentation:

  * Fixed window size = **5000 samples**
* Noise handling (conceptual from paper):

  * Baseline drift removal
  * Filtering

---

### 2️⃣ Feature Extraction (1D-CNN)

* Extracts spatial features like:

  * QRS complex
  * ST-segment
  * T-wave abnormalities

---

### 3️⃣ Temporal Learning (BiLSTM)

* Learns:

  * Heartbeat sequence
  * Time dependencies
* Processes signal **forward + backward**

---

### 4️⃣ Classification

* Fully connected layers
* Sigmoid activation
* Output:

```text
0 → Non-MI
1 → MI
```

---

## 🧠 Model Architecture

```
Input (5000 x 12 ECG signal)
        ↓
Conv1D → BatchNorm → ReLU → MaxPool → Dropout
        ↓
Bidirectional LSTM
        ↓
Dense Layer → Dropout
        ↓
Output Layer (Sigmoid)
```

---

## 🚀 How to Run (Step-by-Step)

### Step 1: Clone Repo

```bash
git clone https://github.com/your-username/mi-detection-ecg.git
cd mi-detection-ecg
```

### Step 2: Install Requirements

```bash
pip install wfdb numpy tensorflow scikit-learn matplotlib
```

### Step 3: Download Dataset

```bash
wget -r -N -c -np https://physionet.org/files/ptbdb/1.0.0/
```

### Step 4: Train Model

```bash
python src/train.py
```

### Step 5: Evaluate Model

```bash
python src/evaluate.py
```

---

## 📈 Results (From IEEE Paper)

* **Accuracy:** 98.74%
* **Precision:** 97.85%
* **Recall:** 98.32%
* **Specificity:** 99.05%
* **F1-Score:** 98.08%

---

## 📊 Evaluation Metrics Used

* Confusion Matrix
* Classification Report
* Accuracy & Loss curves
* ROC Curve & AUC
* Threshold tuning

---

## 🧪 Testing Approach

* Patient-level splitting (Train / Validation / Test)
* Segment-level predictions
* Final prediction:

```text
Average probability of segments → Final decision
```

---

## 🧩 Key Features of Project

✔ Uses **real medical ECG dataset**
✔ Combines **CNN + BiLSTM (Hybrid Model)**
✔ Handles **multi-lead ECG (12 leads)**
✔ Uses **patient-level validation (important in healthcare)**
✔ Supports **threshold tuning for better accuracy**

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* WFDB
* Scikit-learn
* Matplotlib
* Google Colab

---

## 👨‍💻 Authors

* **Dasari Chandu**
* **Parimi Satwika**
* **K. Antony Kumar**

Vel Tech Rangarajan Dr. Sagunthala R&D Institute of Science and Technology, Chennai

---

## 📌 Conclusion

This project demonstrates that combining:

* **CNN (spatial learning)**
* **BiLSTM (temporal learning)**

provides a powerful solution for **automatic myocardial infarction detection**.

The system can be extended for:

* Real-time monitoring
* IoT healthcare devices
* Clinical decision support systems

---

## 🔮 Future Scope

* Deploy as **web/app interface**
* Integrate with **IoT ECG devices**
* Use **Transformer models**
* Improve **AUC score with better tuning**

---

## 📎 References

Based on IEEE research paper and PhysioNet dataset.


