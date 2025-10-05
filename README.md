# HOPE: Hybrid Optimized Parallel Encoding for Depression Symptom Detection

This repository contains the official implementation of **HOPE** (Hybrid Optimized Parallel Encoding), a novel framework for detecting and ranking depression symptoms in social media text.

## Methodology Overview

HOPE addresses the inherent challenges through a three-component architecture:
![Model Architecture](figures/model.png)

### 1. Supervised Symptom Extractor

Leverages **DepRoBERTa** to capture depressive patterns from labeled training data and generate initial relevance scores:

- **Base Model**: DepRoBERTa, a RoBERTa model pre-trained on mental health forum data
- **Task**: Multi-label sentence classification for depression symptom categories
- **Output**: Initial symptom-sentence relevance scores from supervised learning

### 2. Unsupervised Intrinsic Semantic Clustering

Employs few-shot-enhanced symptom-initialized K-means clustering with paired-stream BERT embeddings to extract intrinsic semantic relationships directly from unlabeled testing inputs:

- **Paired-Stream Embeddings**: Combines `nomic-embed-text-v1.5` and `modernbert-embed-base` for robust semantic representations
- **Symptom Initialization**: Cluster centroids initialized using BDI-II symptom descriptions
- **Few-Shot Enhancement**: Incorporates few-shot learning to improve cluster quality
- **Output**: Unsupervised symptom-sentence similarity scores based on semantic clustering

### 3. Optimized Hybrid Semantic Fusion

Leverages the dual-view of supervised and unsupervised scores via a learnable MLP module:

- **Learnable Fusion**: MLP-based fusion mechanism that learns optimal combination of both views
- **Complementary Integration**: Exploits complementary strengths of supervised patterns and unsupervised semantics
- **Output**: Final unified relevance scores yielding more stable and task-agnostic predictions



## Repository Structure

```
.
├── clustering/           # Unsupervised module: embedding, K-means, filtering
├── data/                 # Raw and processed data
├── data_processing/      # Scripts for data loading and preparation
├── extractor/            # Supervised module: model, training, inference
├── figures/              # Model diagrams and plots
├── utils/                # Utility scripts and constants
├── evaluate.py           # Script for evaluating model predictions
├── fusion.py             # Script for fusing supervised and unsupervised scores
├── requirements.txt      # Python dependencies
└── README.md             # This file
```