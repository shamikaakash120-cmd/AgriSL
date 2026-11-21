# AgriSL - Complete Run & Setup Guide

This guide contains everything required to run the AgriSL platform on any computer (for development or evaluation).

---

## 🏛 Architecture Overview

AgriSL consists of 3 services running together:
1. **ML Microservice (Python FastAPI)**: Port `8001`
   - Serves custom-trained MobileNetV2 leaf disease classifier (35 classes across 7 crops).
   - Computes feature embedding cosine similarity against 35 centroids (`ood_stats.npz`) for out-of-distribution (unsupported plant) detection.
   - Detects crop mismatches (e.g. user selected Banana but uploaded Tea).
2. **Backend API (Node.js Express)**: Port `5000`
   - Handles authentication, database persistence (MySQL), role checks, and orchestrates disease detection.
3. **Frontend Application (React + Vite)**: Port `5173`
   - Bilingual user interface (English & Sinhala).

---

## 🚀 Step-by-Step Launch Guide

### 1. Terminal 1: ML Microservice (FastAPI on Port 8001)

All model weights (`disease_model.keras`) and centroid statistics (`ood_stats.npz`) are included directly in `ml-service/model/` inside this repository. No dataset download or retraining is required.

```powershell
# Navigate to the ML microservice directory
cd ml-service

# Create virtual environment (first time only)
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
# venv\Scripts\activate.bat
# Linux/macOS:
# source venv/bin/activate

# Install dependencies (first time only)
pip install -r requirements.txt

# Start the ML service on port 8001
uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

**Health check:**
```powershell
curl http://127.0.0.1:8001/health
# Expected output: {"status":"ok","service":"AgriSL ML Service","model_loaded":true,"ood_centroids_loaded":true}
```

---

### 2. Terminal 2: Backend API (Express on Port 5000)

```powershell
# Navigate to the server directory
cd server

# Install dependencies (first time only)
npm install

# Run database migrations and seed demo data (first time only)
npm run migrate

# Start the Express server in development mode
npm run dev
```

**Health check:**
```powershell
curl http://localhost:5000/api/health
# Expected output: {"status":"ok","service":"AgriSL API"}
```

---

### 3. Terminal 3: Frontend Application (React on Port 5173)

```powershell
# Navigate to the client directory
cd client

# Install dependencies (first time only)
npm install

# Start the Vite development server
npm run dev
```

Open your browser and navigate to:
👉 **`http://localhost:5173`**

---

## 🌿 Machine Learning Component Details

### 1. 35 Disease Classes & 7 Supported Crops
The model classifies leaves into strictly 35 classes across 7 agricultural crops:
- **Banana (කෙසෙල්)**: `Banana_Cordana`, `Banana_Healthy`, `Banana_Panama`, `Banana_Sigatoka`
- **Chilli / Bell Pepper (මිරිස්)**: `Bell_Pepper_Bacterial_Spot`, `Bell_Pepper_Healthy`
- **Corn / Maize (ඉරිඟු)**: `Corn_Common_Rust`, `Corn_Gray_Leaf_Spot`, `Corn_Healthy`, `Corn_Northern_Leaf_Blight`
- **Potato (අර්තාපල්)**: `Potato_Early_Blight`, `Potato_Healthy`, `Potato_Late_Blight`
- **Rice (වී)**: `Rice_Brown_Spot`, `Rice_Healthy`, `Rice_Hispa`, `Rice_Leaf_Blast`, `Rice_Neck_Blast`
- **Tea (තේ)**: `Tea_Algal_Leaf_Spot`, `Tea_Anthracnose`, `Tea_Bird_Eye_Spot`, `Tea_Brown_Blight`, `Tea_Healthy`, `Tea_Red_Leaf_Spot`, `Tea_White_Spot`, `Tea_helopeltis`
- **Tomato (තක්කාලි)**: `Tomato_Bacterial_Spot`, `Tomato_Early_Blight`, `Tomato_Healthy`, `Tomato_Late_Blight`, `Tomato_Leaf_Mold`, `Tomato_Septoria_Leaf_Spot`, `Tomato_Spider_Mites`, `Tomato_Target_Spot`, `Tomato_Yellow_Leaf_Curl_Virus`

### 2. Decision Logic & Scenarios
- **Scenario A: Genuine Match (`status: "ok"`)**
  The user selects a crop (e.g. *Banana*) and uploads a leaf from that crop. The model outputs the specific disease class with its true confidence percentage.
- **Scenario B: Crop Mismatch (`status: "crop_mismatch"`)**
  The user selects a crop (e.g. *Banana*) but uploads a different supported leaf (e.g. *Tea*). The system identifies the true disease (`Tea_helopeltis`), returns `crop_mismatch: true`, and the UI displays a warning banner informing the user that the uploaded leaf is Tea and not to apply Banana treatments.
- **Scenario C: Unsupported Plant (`status: "unsupported_crop"`)**
  The user uploads an untrained leaf (e.g. *Mango, Papaya, Coconut*). The embedding similarity falls below `OOD_THRESHOLD` (`0.45`). The system returns `class_name: null` and `confidence: null` with an explanatory message, preventing false diagnoses or unverified pesticide usage.

### 3. Model Files Included in Git
- `ml-service/model/disease_model.keras`: Trained MobileNetV2 neural network (~23.6 MB).
- `ml-service/model/class_names.json`: The 35 class label definitions.
- `ml-service/model/ood_stats.npz`: 35 per-class 1280-D centroid embeddings for out-of-distribution detection (~163 KB).
- `ml-service/build_ood_stats.py`: Utility script to rebuild centroids from raw training data if desired.
- `ml-service/calibrate_ood.py`: Calibration script for evaluating supported vs unsupported score distributions.
- `ml-service/test_accuracy.py`: Verification script to evaluate classification accuracy.
