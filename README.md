# 🏦 Anti-Money Laundering (AML) Detection Using Machine Learning

> COMM074 Practical Business Analytics | Group 19  
> IBM Synthetic Financial Dataset (HI-Small Variant) | University of Surrey

---

## 📖 Overview

This project develops a scalable machine learning pipeline to detect money laundering transactions within a highly imbalanced financial dataset. Following the **CRISP-DM** methodology, the team built and evaluated eight predictive models across four architectural families, comparing linear baselines against advanced ensemble and deep learning approaches.

The core challenge: fraudulent transactions represent only **~0.10%** of all records — making this an extreme class imbalance problem where traditional accuracy is meaningless.

---

## 👥 Team

| Name | URN | Contribution |
|------|-----|--------------|
| Piyush Gupta | 6950073 | ElasticNet Logistic Regression & LightGBM |
| Parth Singh Jhala | 6942760 | EDA, Vanilla Logistic Regression & XGBoost |
| Akash Chohan | 6949904 | L1 Logistic Regression & Tuned Random Forest |
| Nagashri Raghunath | 6948579 | L2 Logistic Regression & MLP Neural Network |

---

## 📁 Repository Structure

```
├── Group_Notebook_1_Data_Prep.ipynb        # EDA, preprocessing, SMOTE balancing
├── Group_Notebook_2_Visualisation.ipynb    # Cross-group dashboard & comparison
├── Piyush_Modelling_Notebook_final.ipynb   # ElasticNet LR & LightGBM
├── Parth_Modelling_Notebook_final.ipynb    # Vanilla LR & XGBoost
├── Akash_Modelling_CW_F.ipynb              # L1 LR & Tuned Random Forest
├── Nagashri_Modeling.ipynb                 # L2 LR & MLP Neural Network
├── cross_group_comparison_final.csv        # Aggregated results for dashboard
├── COMM074_Group_19_Final_Report.pdf       # Full project report
└── README.md                               # This file
```

---

## 📦 Dataset

The project uses the **IBM Transactions for Anti-Money Laundering (AML) — HI-Small** dataset, available on Kaggle. Due to its size (~1.3 GB processed), the dataset is **not included** in this repository.

To reproduce the data pipeline, run `Group_Notebook_1_Data_Prep.ipynb` which will automatically download and process the dataset. It will:
1. Download the IBM AML HI-Small dataset from Kaggle
2. Perform EDA and feature engineering
3. Apply StandardScaler and one-hot encoding
4. Apply **SMOTE** (10% ratio) to balance the training set
5. Export `.parquet` and `.npy` files for the modelling notebooks

---

## 🧪 Models & Results

| Model | Author | Recall | AUPRC | ROC-AUC |
|-------|--------|--------|-------|---------|
| LR (ElasticNet) | Piyush | 0.2986 | 0.0232 | 0.5326 |
| **LightGBM** | **Piyush** | **0.6754** | **0.2405** | **0.9683** |
| LR (L1 Lasso) | Akash | 0.1400 | 0.0100 | 0.9140 |
| Tuned Random Forest | Akash | 0.3100 | 0.1700 | 0.9134 |
| LR (No Penalty) | Parth | 0.1768 | 0.0286 | 0.8941 |
| XGBoost (Tuned, threshold=0.93) | Parth | 0.8039 | 0.1962 | 0.9676 |
| LR (L2 Ridge) | Nagashri | 0.0531 | 0.0218 | 0.8985 |
| MLP (128, 64) | Nagashri | 0.1787 | 0.0272 | 0.8868 |

> **Primary metric: Recall & AUPRC** — Accuracy is misleading due to extreme class imbalance (~0.10% fraud rate).

---

## 🏆 Key Findings

- **Linear models consistently underperform** — Recall ranged from 5.3% to 29.8% across all LR variants
- **Gradient boosting dominates** — LightGBM achieved 67.5% Recall with the highest AUPRC (0.2405)
- **XGBoost with threshold tuning** (0.93) is the recommended deployment model — 80.3% Recall with 55% fewer false positives vs default threshold
- **Top fraud predictors**: Payment format (Cheque, Credit Card), Amount Received/Paid, and Bank IDs
- **Neural networks underperform tree-based models** on structured tabular financial data

---

## 📓 View Notebooks

> GitHub may fail to render notebooks. Use the links below to view them properly via **nbviewer**:

| Notebook | nbviewer Link |
|----------|--------------|
| Data Preparation & EDA | [View](https://nbviewer.org/github/ParthSJ/aml-detection-classifier/blob/main/Group_Notebook_1_Data_Prep.ipynb) |
| Visualisation Dashboard | [View](https://nbviewer.org/github/ParthSJ/aml-detection-classifier/blob/main/Group_Notebook_2_Visualisation.ipynb) |
| Piyush — LightGBM & ElasticNet LR | [View](https://nbviewer.org/github/ParthSJ/aml-detection-classifier/blob/main/Piyush_Modelling_Notebook_final.ipynb) |
| Parth — XGBoost & Vanilla LR | [View](https://nbviewer.org/github/ParthSJ/aml-detection-classifier/blob/main/Parth_Modelling_Notebook_final.ipynb) |
| Akash — Random Forest & L1 LR | [View](https://nbviewer.org/github/ParthSJ/aml-detection-classifier/blob/main/Akash_Modelling_CW_F.ipynb) |
| Nagashri — MLP & L2 LR | [View](https://nbviewer.org/github/ParthSJ/aml-detection-classifier/blob/main/Nagashri_Modeling.ipynb) |

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run data preparation first:
```bash
jupyter notebook Group_Notebook_1_Data_Prep.ipynb
```
4. Then run any individual modelling notebook

---

## 📊 Methodology

The project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework:

1. **Business Understanding** — AML detection framed as imbalanced binary classification
2. **Data Understanding** — EDA on 5M+ transactions, 42 features
3. **Data Preparation** — One-hot encoding, StandardScaler, SMOTE (10% ratio)
4. **Modelling** — 8 models across 4 families trained on same data artifacts
5. **Evaluation** — Recall, Precision, F1, AUPRC as primary metrics
6. **Deployment** — Strategic recommendations for compliance departments

---

## 📄 License

This project was developed for academic purposes as part of the COMM074 Practical Business Analytics module at the University of Surrey.
