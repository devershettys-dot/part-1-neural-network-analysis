# Part 1 — Customer Churn Neural Network Analysis

## Overview
Feed-forward neural network built to predict customer churn using `customer_churn_nn.csv`.  
Target variable: `churn` (1 = churned, 0 = retained).

## Repository Structure
```
part-1-neural-network-analysis/
├── README.md
├── notebook.ipynb            ← main notebook (all 6 tasks)
├── requirements.txt
└── results/
    ├── model_comparison_table.csv   ← Task 5 experiment results
    ├── model_comparison_table.png   ← Task 5 visual table
    └── evaluation_outputs.png       ← Task 4 confusion matrix
```

## Tasks Covered
| Task | Description |
|------|-------------|
| 1 | Dataset understanding — shape, types, missing values, class distribution |
| 2 | Preprocessing — StandardScaler, OneHotEncoder, stratified train/test split |
| 3 | Neural network — 2 hidden layers, ReLU, Dropout, Sigmoid output |
| 4 | Training & evaluation — class_weight, EarlyStopping, ROC-AUC, confusion matrix |
| 5 | Hyperparameter experiments — 4 configs compared across Recall, F1, ROC-AUC |
| 6 | Final reflection — weights/biases, activations, learning rate effects, fit diagnosis |

## Key Finding
Severe class imbalance (98.5% retained, 1.5% churned). Best model: **Tanh + Batch 64**  
(ROC-AUC = 0.93, Churn Recall = 1.0).

## Setup
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```
