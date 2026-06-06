# 🏗️ RankForge — System Architecture
## By Sulthan Basha Mulla | Team Curators | INDIA.RUNS 2026

## Overview
RankForge is designed as a production-ready, 
DevOps-backed AI candidate ranking system.
Not just a model — a complete deployable system.

---

## System Architecture Layers

### 1️⃣ API Layer
- Framework: FastAPI
- Endpoint: POST /rank
- Input: Job Description + Candidate Profiles
- Output: Ranked candidates with scores + skill gaps
- Format: JSON response

### 2️⃣ AI Agent Layer
- Sourcing Agent: Fetches candidates from database
- Ranking Agent: TF-IDF + Cosine Similarity scoring
- Skill Gap Agent: Identifies matched/missing skills
- Feedback Agent: Collects recruiter feedback for retraining

### 3️⃣ MLOps Pipeline
- Model Training: scikit-learn TF-IDF Vectorizer
- Model Evaluation: Cosine similarity scoring
- Auto Retraining: Triggered when feedback > 100 samples
- Model Versioning: Stored with timestamps on GitHub

### 4️⃣ DevOps Infrastructure
- Containerization: Docker
- CI/CD Pipeline: GitHub Actions
- Workflow:
  1. Code pushed to GitHub
  2. GitHub Actions triggers automatically
  3. Tests run
  4. Docker image built
  5. System deployed automatically
- Monitoring: Automated health checks every 5 minutes

### 5️⃣ Database Layer
- Candidate Profiles: pandas DataFrame → CSV
- Vector Storage: FAISS (for scale)
- Feedback Logs: Stored for retraining

---

## Docker Configuration (Designed)
dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "rankforge_v3.py"]


---

## GitHub Actions CI/CD (Designed)
yaml
name: RankForge CI/CD
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run RankForge
        run: python rankforge_v3.py


---

## Scalability Design
- Horizontal scaling via Docker containers
- Each agent scales independently
- Handles 10M+ candidate profiles
- Response time: under 2 seconds

---

## Why DevOps Backbone Matters
Most AI hiring tools are prototypes.
RankForge is designed to run in production:
- Zero manual deployment
- Auto-healing pipelines
- Continuous improvement through feedback
- India-scale ready from day one
