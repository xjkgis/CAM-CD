# CAM-CD: A Change-Aware Mamba Framework for Remote Sensing Change Detection

This is the official implementation of **CAM-CD**, a Mamba-based framework designed for high-precision remote sensing change detection.

## 🌟 Highlights
- **Change-Aware Mamba:** Unlike traditional uniform scanning, CAM-CD aligns state transitions with sparse and structured changes.
- **CAAS Module:** A Change-Aware Adaptive Scanning mechanism that uses spatial priors to focus on change-relevant areas and suppress noise.
- **S2FM:** A Selective Scanning Fusion Module that treats bi-temporal interaction as a path-dependent process with recursive "change memory."

## 🛠️ Project Structure
- `models/`: Contains the core architecture of CAM-CD (CAAS and S2FM).
- `utils/`: Utility functions for data loading and processing.
- `train_final.py`: The main script for training the model.
- `test_final.py`: The script for evaluation and generating change maps.

## 🚀 Getting Started

### 1. Requirements
- Python 3.10+
- PyTorch 2.1.2
- CUDA 11.8
- (Dependencies for Mamba/Selective Scan)
