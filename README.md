# ecg-hrv-analysis
Thesis project - ECG and HRV analysis using NeuroKit2 and HRV-analysis 

# 🫀 ECG & HRV Analysis Toolkit

This repository contains Python code developed for my undergraduate thesis project on cardiovascular health and mood disorders. It processes raw ECG waveform data to extract R-peaks and compute heart rate variability (HRV) features across time and frequency domains.

The analysis pipeline **combines two powerful open-source libraries**:  
- [NeuroKit2](https://github.com/neuropsychology/NeuroKit) – for signal preprocessing and R-peak detection  
- [HRV-analysis](https://github.com/Aura-healthcare/hrvanalysis) – for HRV feature extraction  

---

## 📦 What's Included

- `main.py` – The full script for ECG processing and HRV computation  
- `requirements.txt` – Python dependencies  
- `.gitignore` – Clean version control  
- `LICENSE` – GPL-3.0 license text  

---

## 🧪 Features

✔️ ECG signal loading and preprocessing  
✔️ R-peak detection using `pantompkins1985` algorithm (NeuroKit2)  
✔️ Time-domain and frequency-domain HRV metrics  
✔️ NaN handling and ectopic beat removal  
✔️ Optional memory-efficient processing for large ECG files  
✔️ Sampling rate - 250 Hz in this file but change according to your raw ECG data 

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ecg-hrv-analysis.git
cd ecg-hrv-analysis
