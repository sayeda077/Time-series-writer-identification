# 🧠 Time-Series Writer Identification using LSTM & BiLSTM

This project focuses on identifying writers based on **time-series handwriting data** using deep learning models such as **LSTM** and **BiLSTM**.

---

## 📌 Project Overview

Handwriting contains unique patterns for each individual. In this project, we use **time-series data (pen movement signals)** to build a model that can classify which writer produced a given sequence.

We implemented:
- ✅ LSTM (Long Short-Term Memory)
- ✅ BiLSTM (Bidirectional LSTM)

---

## 📂 Dataset

The dataset consists of **online handwriting signals** collected from multiple writers.

### 🔹 Key Characteristics:
- Total writers: **53**
- Each sample is a **time-series sequence**
- Each timestep contains **13 features** (e.g., pen coordinates, velocity, pressure, etc.)
- Sequence length varies (e.g., 200+ timesteps)

### 🔹 Data Format:
The dataset is stored in `.pkl` (pickle) files:

| File Name | Description |
|----------|-------------|
| `all_x_dat_train_imu.pkl` | Training input sequences |
| `train_ids.pkl` | Training labels (writer IDs) |
| `all_x_dat_val_imu.pkl` | Validation input sequences |
| `val_ids.pkl` | Validation labels |

### 🔹 Example:
- Input shape before processing: `(variable_length, 13)`
- Example: `(220, 13)`
- Label: Writer ID (e.g., `1001`, `1002`, etc.)

---

## ⚙️ Preprocessing

Since sequences have different lengths:

- ✂️ **Truncation** → Long sequences are cut  
- ➕ **Padding** → Short sequences are padded with zeros  

Final fixed input shape:
