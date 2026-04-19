# 🎯 YouTube Viewer Sentiment Analyzer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-Classifier-brightgreen?style=for-the-badge&logo=leaflet&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-REST%20API-black?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/MLflow-Tracking-blue?style=for-the-badge&logo=mlflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/DVC-Pipeline-945DD6?style=for-the-badge&logo=dvc&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/AWS-ECR%20%7C%20EC2-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white"/>
  <img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white"/>
</p>

<p align="center">
  <b>An end-to-end production-grade ML system that classifies YouTube comments into Positive, Neutral, and Negative sentiments — powered by LightGBM, TF-IDF, Flask, and a Chrome Extension UI.</b>
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [ML Pipeline](#-ml-pipeline)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Chrome Extension](#-chrome-extension)
- [Experiment Tracking](#-experiment-tracking-with-mlflow)
- [CI/CD & Deployment](#-cicd--aws-deployment)
- [Hyperparameters](#-model-hyperparameters)

---

## 🧠 Overview

**YouTube Viewer Sentiment Analyzer** is a full-stack, production-ready machine learning system that analyzes the emotional tone of YouTube comment sections in real time. It combines:

- 🔬 **A rigorous ML experimentation pipeline** — 8 experiments across Bag-of-Words, TF-IDF, XGBoost, and LightGBM models with hyperparameter tuning
- ⚙️ **A reproducible DVC pipeline** — from raw data ingestion → preprocessing → model training → evaluation → registration
- 🌐 **A Flask REST API** — serving predictions, word clouds, pie charts, and sentiment trend graphs
- 🧩 **A Chrome Extension** — letting users analyze any YouTube video's comment section directly from their browser
- ☁️ **Full AWS deployment** — containerized with Docker, stored in ECR, deployed on EC2, with automated CI/CD via GitHub Actions

---

## 🏗️ Architecture

```
YouTube Comments (via YouTube Data API v3)
         │
         ▼
┌─────────────────────────┐
│   Chrome Extension UI   │  ← popup.js fetches comments & calls Flask API
└────────────┬────────────┘
             │  HTTP POST
             ▼
┌─────────────────────────┐
│     Flask REST API      │  ← Preprocessing · Prediction · Visualization
│   (app.py / flask_app)  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  LightGBM Classifier    │  ← Trained on TF-IDF (1,3)-grams · 1000 features
│  + TF-IDF Vectorizer    │
└────────────┬────────────┘
             │
             ▼
    [Positive / Neutral / Negative]
```

---

## 🔁 ML Pipeline

The pipeline is fully orchestrated with **DVC** and tracked with **MLflow**.

```
data_ingestion → data_preprocessing → model_building → model_evaluation → model_registration
```

| Stage | Script | Output |
|---|---|---|
| `data_ingestion` | `src/data/data_ingestion.py` | `data/raw/` (train/test CSV) |
| `data_preprocessing` | `src/data/data_preprocessing.py` | `data/interim/` (cleaned CSV) |
| `model_building` | `src/model/model_building.py` | `lgbm_model.pkl`, `tfidf_vectorizer.pkl` |
| `model_evaluation` | `src/model/model_evaluation.py` | MLflow metrics, confusion matrix |
| `model_registration` | `src/model/register_model.py` | Registered model in MLflow registry |

---

## 📂 Project Structure

```
YouTube-Viewer-Sentiment/
│
├── src/
│   ├── data/
│   │   ├── data_ingestion.py          # Fetches & splits raw data
│   │   └── data_preprocessing.py      # Cleans, lemmatizes, removes stopwords
│   └── model/
│       ├── model_building.py          # TF-IDF + LightGBM training
│       ├── model_evaluation.py        # MLflow logging, confusion matrix
│       └── register_model.py          # Pushes model to MLflow registry
│
├── flask_app/
│   └── app.py                         # Standalone Flask app (for Docker)
├── app.py                             # Root Flask entry point
│
├── yt-chrome-plugin-frontend/
│   ├── manifest.json                  # Chrome extension manifest
│   ├── popup.html                     # Extension UI
│   └── popup.js                       # Fetches YouTube comments & calls API
│
├── notebooks/                         # 8 experiment notebooks (EDA → LightGBM HPT)
├── dvc.yaml                           # DVC pipeline definition
├── dvc.lock                           # Locked pipeline state
├── params.yaml                        # Hyperparameters
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup
├── Dockerfile                         # Container definition
├── .github/workflows/cicd.yaml        # GitHub Actions CI/CD
├── lgbm_model.pkl                     # Trained model artifact
└── tfidf_vectorizer.pkl               # Fitted TF-IDF vectorizer
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/YouTube-Viewer-Sentiment.git
cd YouTube-Viewer-Sentiment
```

### 2. Create and activate environment

```bash
conda create -n youtube python=3.11 -y
conda activate youtube
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the DVC pipeline

```bash
dvc init        # (only if starting fresh)
dvc repro       # Reproduces the full pipeline end-to-end
dvc dag         # Visualize the pipeline DAG
```

### 5. Start the Flask API

```bash
python app.py
# Running on http://0.0.0.0:8080
```

---

## 📡 API Reference

Base URL: `http://localhost:8080`

### `POST /predict`
Classify a list of comments into sentiments.

**Request:**
```json
{
  "comments": [
    "This video is awesome! I loved it a lot.",
    "Very bad explanation. Poor video."
  ]
}
```

**Response:**
```json
[
  { "comment": "This video is awesome! I loved it a lot.", "sentiment": 1 },
  { "comment": "Very bad explanation. Poor video.", "sentiment": -1 }
]
```

> Sentiment values: `1` = Positive · `0` = Neutral · `-1` = Negative

---

### `POST /predict_with_timestamps`
Classify comments and return them with their timestamps for trend analysis.

**Request:**
```json
{
  "comments": [
    { "text": "Great video!", "timestamp": "2024-01-15T10:00:00Z" }
  ]
}
```

---

### `POST /generate_chart`
Returns a **PNG pie chart** of sentiment distribution.

```json
{ "sentiment_counts": { "1": 45, "0": 30, "-1": 25 } }
```

---

### `POST /generate_wordcloud`
Returns a **PNG word cloud** generated from preprocessed comment text.

```json
{ "comments": ["great video", "loved the content", "amazing explanation"] }
```

---

### `POST /generate_trend_graph`
Returns a **PNG line chart** of monthly sentiment percentages over time.

---

## 🧩 Chrome Extension

The Chrome Extension lets you analyze any YouTube video's comment section directly in your browser without leaving the page.

**Setup:**
1. Open Chrome and navigate to `chrome://extensions`
2. Enable **Developer Mode** (top-right toggle)
3. Click **Load unpacked** → select the `yt-chrome-plugin-frontend/` folder
4. Open any YouTube video and click the extension icon

**Get a YouTube Data API v3 Key:**
> Follow this guide: https://www.youtube.com/watch?v=i_FdiQMwKiw
> Then set your API key inside `popup.js`.

---

## 📊 Experiment Tracking with MLflow

All training runs are tracked with **MLflow** — metrics, parameters, artifacts, and the model registry are all logged automatically.

| Metric | Value |
|---|---|
| Model | LightGBM (multiclass) |
| Feature Extraction | TF-IDF (1,3)-gram · 1000 features |
| Experiment Name | `dvc-pipeline-runs` |
| Logged Metrics | Precision · Recall · F1-score per class |
| Logged Artifacts | Confusion matrix · TF-IDF vectorizer |

**Tracked experiments (notebooks):**

| # | Experiment | Description |
|---|---|---|
| 1 | Preprocessing & EDA | Data cleaning, label distribution, language filtering |
| 2 | Baseline Model | Logistic Regression on Bag-of-Words |
| 3 | Experiment 2 | BoW vs TF-IDF comparison |
| 4 | Experiment 3 | TF-IDF (1,3)-gram with max features tuning |
| 5 | Experiment 4 | Handling class imbalance (SMOTE / class weights) |
| 6 | Experiment 5 | XGBoost with hyperparameter tuning |
| 7 | Experiment 6 | LightGBM with detailed HPT (best model ✅) |
| 8 | Stacking | Ensemble methods exploration |

---

## 🏋️ Model Hyperparameters

```yaml
model_building:
  ngram_range: [1, 3]       # Unigram to Trigram
  max_features: 1000        # TF-IDF vocabulary size
  learning_rate: 0.09       # LightGBM learning rate
  max_depth: 20             # Maximum tree depth
  n_estimators: 367         # Number of boosting rounds

data_ingestion:
  test_size: 0.20           # 80/20 train-test split
```

---

## ☁️ CI/CD & AWS Deployment

Every push to `main` triggers a fully automated 3-stage GitHub Actions pipeline:

```
CI (Lint & Test) → CD (Build & Push to ECR) → Deploy (Pull & Run on EC2)
```

### AWS Setup Guide

#### 1. Configure AWS CLI
```bash
aws configure
# Enter: AWS Access Key ID, Secret, Region (us-east-1), Output format
```

#### 2. Create IAM User with Required Policies
- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2FullAccess`

#### 3. Create ECR Repository
```bash
# Save the URI after creation, e.g.:
# 315865595366.dkr.ecr.us-east-1.amazonaws.com/youtube
```

#### 4. Launch EC2 Instance (Ubuntu) & Install Docker
```bash
sudo apt-get update -y && sudo apt-get upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

#### 5. Configure EC2 as Self-Hosted GitHub Runner
```
GitHub Repo → Settings → Actions → Runners → New self-hosted runner → Follow OS-specific steps
```

#### 6. Set GitHub Secrets

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | Your IAM Access Key |
| `AWS_SECRET_ACCESS_KEY` | Your IAM Secret Key |
| `AWS_REGION` | `us-east-1` |
| `AWS_ECR_LOGIN_URI` | e.g. `315865595366.dkr.ecr.us-east-1.amazonaws.com` |
| `ECR_REPOSITORY_NAME` | e.g. `youtube` |

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ — LightGBM · TF-IDF · Flask · DVC · MLflow · Docker · AWS · GitHub Actions
</p>
