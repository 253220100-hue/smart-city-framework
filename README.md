# Smart City Computational Framework

## Overview
This repository provides a reproducible computational framework for decision support in smart cities. The framework integrates heterogeneous urban data, hybrid AI models, topology-aware validation, and multi-objective evaluation.

## Features
- End-to-end pipeline for urban data processing
- Hybrid AI modeling (forecasting, RL, fuzzy logic, Bayesian inference)
- Topology-aware validation (connectivity, cycles, robustness)
- Scenario generation under uncertainty
- Multi-objective optimization (Pareto analysis)

## Architecture
The framework is organized into five phases:

1. Data ingestion and normalization  
2. Representation learning  
3. Hybrid modeling and scenario generation  
4. Topology-aware validation  
5. Multi-objective evaluation

## Data

The repository includes a small synthetic dataset (data/sample.csv) for demonstration purposes. 

Users can replace this dataset with real-world urban data sources.

## Results

Due to the large file size of high-resolution TIFF images (600 DPI) required for publication, lightweight JPEG versions are included in this repository for quick visualization and verification purposes.

These images demonstrate the successful execution of the computational pipeline and the generation of model outputs.

The original high-resolution TIFF figures, suitable for publication standards (IGI Global), are generated automatically by the code and can be reproduced by executing:

python run.py

All figures are stored in:
results/figures/

This ensures full reproducibility of the results while maintaining repository efficiency.

## Installation

```bash
git clone https://github.com/your-username/smart-city-computational-framework.git
cd smart-city-computational-framework
pip install -r requirements.txt
python run.py
