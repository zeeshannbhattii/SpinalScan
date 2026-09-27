# SpinalScan — Clinical Spine & Lumbar Radiology Diagnostic Platform

A clean, modern, and professional diagnostic platform for analyzing **Spine Vertebrae X-Rays** and **Lumbar Spine MRI Scans**.

---

## 🌐 Live Deployment on Vercel (`vercel.com`)

The application is pre-configured and optimized for **Vercel Serverless Functions** (`@vercel/python`):

- **Target URL:** `https://spinalscan.vercel.app`
- **Configuration:** [vercel.json](file:///c:/Users/HP/OneDrive/Desktop/SpinalScan/vercel.json)
- **Serverless Entrypoint:** [api/index.py](file:///c:/Users/HP/OneDrive/Desktop/SpinalScan/api/index.py)
- **Deployment Bundle:** `SpinalScan-Vercel-Deployment.zip`

### 3-Step 1-Click Deployment via GitHub to Vercel:

1. **Create Repository:**
   - Go to [github.com/new](https://github.com/new).
   - Enter Repository name: **`SpinalScan`**.
   - Set to Public and click **Create repository**.

2. **Upload Project Files (Important: Upload Extracted Files, NOT the .zip file itself):**
   - In GitHub, click **Upload files** (or use Git CLI / GitHub Desktop).
   - Drag and drop all the project files and folders:
     - `api/`
     - `sample_images/`
     - `static/`
     - `templates/`
     - `index.html`
     - `server.py`
     - `predictor.py`
     - `vercel.json`
     - `requirements.txt`
     - `.python-version`
   - Click **Commit changes** to `main`.

3. **Deploy on Vercel:**
   - Go to [vercel.com/new](https://vercel.com/new).
   - Sign in with your GitHub account.
   - Find your **`SpinalScan`** repository and click **Import**.
   - Under **Framework Preset**, leave as **Other** (Vercel automatically detects `vercel.json` and `@vercel/python`).
   - Click **Deploy**.
   - Your website will immediately go live at:
     👉 **`https://spinalscan.vercel.app`**

---

## 🚀 Local Launch (Open in Chrome)

Double-click **`run_website.bat`** or run in PowerShell:
```powershell
python server.py
```
Open your browser at:
👉 **[http://localhost:5000](http://localhost:5000)**

---

## 🌟 Clinical Modalities & Scope

1. **Spine X-Ray:**
   - Detects **Normal Alignment**, **Scoliosis**, and **Spondylolisthesis**.
   - Generates confidence score, percentage breakdown, and anatomical localization heatmap.

2. **Lumbar Spine MRI:**
   - Detects **Herniated Disc**, **Thecal Sac Stenosis**, and **No Stenosis**.
   - Analyzes sagittal and axial MRI sections.
   - Provides recommended clinical next steps.

3. **Dual-Scan Comprehensive Study:**
   - Evaluates both a Spine X-Ray and a Lumbar MRI scan concurrently.
   - Generates an integrated report comparing both spinal levels.
   - One-click exportable clinical summary report (`.txt`).

---

## 📁 Project Structure

```text
├── vercel.json                           # Vercel routing & serverless build configuration
├── api/
│   └── index.py                          # Vercel Python serverless entrypoint
├── requirements.txt                      # Lightweight dependencies (<250MB)
├── .vercelignore                         # Excludes bulky raw training files
├── .gitignore                            # Clean repo settings
├── server.py                             # Production Flask web server & REST API
├── templates/
│   └── index.html                        # Professional medical blue UI (SpinalScan)
├── static/
│   ├── css/style.css                     # Medical blue theme & clean typography
│   └── js/app.js                         # Dynamic scan analysis & reports
├── predictor.py                          # Multi-runtime inference (TFLite & Keras)
├── models/
│   ├── spine_xray_classifier.tflite      # Optimized spine model (16 MB)
│   ├── lumbar_mri_efficientnet.tflite    # Calibrated lumbar MRI model (16 MB)
│   ├── spine_xray_classifier_final.keras # Keras spine model
│   └── lumbar_mri_efficientnet_final.keras# Keras lumbar model
├── sample_images/                        # Verified clinical sample scans
├── SpinalScan-Vercel-Deployment.zip      # 30 MB pre-packaged deployment bundle
├── run_website.bat                       # Local 1-click launcher
└── README.md                             # Documentation
```

---

## 🔒 Clinical Disclaimer
SpinalScan is designed as a medical diagnostic decision-support system. All algorithmic analyses must be clinically validated by a board-certified radiologist or orthopedic specialist.
