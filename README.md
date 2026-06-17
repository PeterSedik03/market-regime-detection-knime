# Market Regime Detection and Next-Day Direction Forecasting on Multi-Asset Financial Time Series

**Peter Emad Sami Sedik** — MSc Data Science, Università di Milano-Bicocca  
Machine Learning course project — Prof. Fabio Stella — A.Y. 2025/2026

---

## Overview

This project applies unsupervised and supervised machine learning techniques to a multi-asset financial time series covering four instruments — **DXY, Gold, Silver, and S&P 500** — over the trading period March 2024 to May 2026.

Two problems are addressed:

1. **Market regime detection** via Ward hierarchical clustering and K-Medoids partitioning (K = 3), validated through internal indices, external comparison, and a Monte Carlo significance test.
2. **Binary next-day direction forecasting** (UP/DOWN) using Decision Tree, Random Forest, Naive Bayes, Logistic Regression, and an Attribute-Selected J48, evaluated under holdout, 10-fold cross-validation, Wilson confidence intervals, and paired t-tests.

The entire analysis is implemented in **KNIME Analytics Platform**.

---

## Dataset

`integrated_data.csv` was personally constructed for the Data Management course at Università di Milano-Bicocca. It integrates:

- Daily OHLCV market prices for 4 assets
- Technical indicators: SMA, EMA, RSI, Bollinger Bands, volatility
- News-based sentiment features: sentiment scores, news count, positive/negative ratios

**2,149 observations** across 4 assets (trading period: 27 March 2024 – 15 May 2026).  
Binary target: `direction = UP` if next-day return > 0, else `DOWN`.  
Class distribution: UP 56.0% / DOWN 44.0% → majority-class baseline = **56.0%**.

---

## Methodology

### Preprocessing
CSV Reader → Missing Value (mean imputation) → Sorter → Lag Column (t+1 return) → Rule Engine (UP/DOWN target) → Column Filter → One-to-Many encoding → Z-score normalisation (clustering) + Min-Max normalisation (classification).

### Clustering
- **Ward linkage** dendrogram (Python/scipy) → K = 3 selected
- **K-Medoids** (K = 3, Euclidean distance)
- Validation: Silhouette, Dunn index, Rand, Jaccard, Fowlkes-Mallows, Monte Carlo significance test (1,000 permutations)

### Classification
- **Holdout** 70/30 linear split → Decision Tree, Random Forest, Naive Bayes, Logistic Regression
- **10-fold Cross-Validation** → same classifiers + AttributeSelected J48 (CfsSubsetEval + BestFirst)
- Wilson 95% confidence intervals and paired t-tests

---

## Key Results

### Clustering

| Index | Value |
|-------|-------|
| Silhouette | 0.28 |
| Dunn | 0.021 |
| Rand | 0.497 |
| Jaccard | 0.275 |
| Fowlkes-Mallows | 0.436 |
| Monte Carlo p-value | < 0.001 |

Three statistically significant but weakly separated market regimes identified.

### Classification — 10-fold Cross-Validation

| Classifier | Accuracy ± SD | κ |
|------------|--------------|---|
| Attr. Selected J48 | 0.560 ± 0.033 | 0.000 |
| Logistic Regression | 0.547 ± 0.046 | 0.027 |
| Random Forest | 0.545 ± 0.049 | −0.011 |
| Naive Bayes | 0.523 ± 0.048 | 0.021 |
| Decision Tree | 0.503 ± 0.041 | −0.007 |
| **Majority-class baseline** | **0.560** | — |

CFS feature selection retains only `high` with merit = 0.000, producing a single-leaf J48 tree that always predicts UP — strong empirical support for the **Efficient Market Hypothesis**.

---

## Repository Structure

```
├── integrated_data.csv          # Custom-built multi-asset financial dataset
├── KNIME_Sedik_944958.knwf      # KNIME workflow (import into KNIME 5.x)
└── ML_Report_Sedik_944958.pdf   # Full project report
```

---

## Tools

- **KNIME Analytics Platform 5.8.3** — workflow, preprocessing, clustering, classification
- **Python** (scipy, sklearn, matplotlib) — Ward clustering, Monte Carlo test, Wilson CI
- **Weka nodes** — AttributeSelectedClassifier (CfsSubsetEval + BestFirst)

---

## How to Run

1. Install [KNIME Analytics Platform](https://www.knime.com/downloads) 5.x
2. Import `KNIME_Sedik_944958.knwf` via File → Import KNIME Workflow
3. Update the CSV Reader node path to point to `integrated_data.csv`
4. Execute the workflow section by section (Preprocessing → Clustering → Classification)
