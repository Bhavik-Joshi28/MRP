# MRP
Major Research Project 
# Financial Fraud Detection Project

## Overview

This project focuses on developing an advanced machine learning model to detect financial fraud, particularly in gas station transactions. The approach incorporates time-based features, distance calculations, and geospatial analytics to enhance the model's accuracy and reliability.

## Contents

- `data/` - Directory containing the datasets used for training and testing.
  - `fraud_train.csv` - Training dataset.
  - `fraud_test.csv` - Testing dataset.
  
- `notebooks/` - Jupyter notebooks for exploratory data analysis, model training, and evaluation.
  - `Fraud_Detection_Analysis.ipynb` - Detailed analysis and model evaluation.

- `src/` - Source code for the project.
  - `feature_engineering.py` - Code for feature engineering including time-based and distance features.
  - `model_training.py` - Code for training various machine learning models with different sampling techniques.
  - `visualization.py` - Code for visualizing results and fraud hotspots.

- `results/` - Directory for saving model results and comparison tables.
  - `model_comparison.xlsx` - Excel file with model comparison results.

- `requirements.txt` - List of Python packages required for the project.

- `README.md` - This file.

## Getting Started

### Prerequisites

To run the code and notebooks, you need Python 3.7 or later and the following packages:

- `numpy`
- `pandas`
- `scikit-learn`
- `imblearn`
- `folium`
- `matplotlib`
- `seaborn`
- `openpyxl` (for Excel file handling)

You can install the required packages using `pip`:

```bash
pip install -r requirements.txt



