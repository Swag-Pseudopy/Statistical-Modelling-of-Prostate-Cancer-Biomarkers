# Statistical Modelling of Prostate Cancer Biomarkers

## 📌 Project Overview

This project presents a rigorous statistical analysis of prostate cancer data, aiming to model the **Log Prostate Specific Antigen (`lpsa`)** using various clinical covariates. Prostate cancer is a significant global health challenge, and accurate statistical modeling of biomarkers is crucial for risk stratification and decision-making regarding active surveillance versus surgical intervention.

This analysis compares **Multiple Linear Regression (OLS)** against advanced techniques including **Robust Regression**, **Lasso Regularization**, and **Generalized Additive Models (GAM)** to determine the optimal predictive model.

## 📂 Dataset

The dataset consists of clinical data from **97 men** who were about to undergo a radical prostatectomy. The data is split into a training set ($n=67$) and a testing set ($n=30$).

| Variable | Description |
| :--- | :--- |
| **`lcavol`** | Logarithm of cancer volume. |
| **`lweight`** | Logarithm of prostate weight. |
| **`age`** | Age of the patient. |
| **`lbph`** | Logarithm of benign prostatic hyperplasia amount. |
| **`svi`** | Seminal vesicle invasion (Binary: 0 or 1). |
| **`lcp`** | Logarithm of capsular penetration. |
| **`gleason`** | Gleason score (6-9). |
| **`pgg45`** | Percentage of Gleason scores 4 or 5. |
| **`lpsa`** | **Response Variable**: Log of Prostate Specific Antigen. |
| **`train`** | Logical flag indicating training data. |

## ⚙️ Methodology

The analysis follows a multi-stage statistical framework:

1.  **Exploratory Data Analysis (EDA):** Assessment of correlation structures and linearity using scatterplots and heatmaps.
2.  **Baseline Modeling:** Fitting a full OLS model followed by **Stepwise AIC selection** to induce parsimony.
3.  **Diagnostics:**
      * Normality (Shapiro-Wilk)
      * Heteroscedasticity (Breusch-Pagan)
      * Multicollinearity (VIF)
      * Influence Analysis (Cook's Distance)
4.  **Advanced Modeling:**
      * **Robust Regression (RLM):** Using Huber M-estimation to down-weight influential observations.
      * **Lasso Regression:** For regularization and variable selection (handling multicollinearity).
      * **Generalized Additive Models (GAM):** To capture potential non-linear relationships using smooth splines.
5.  **Validation:** Models were evaluated on the held-out test set using **RMSE** and **$R^2$**.

## 📊 Key Results

The study found that linear structures dominate the data generating process. Despite identifying influential points via Cook's Distance, treating them as outliers (via Robust Regression) reduced predictive performance, suggesting they carried valid biological signals.

**Model Performance on Test Data ($n=30$):**

| Model | Test RMSE | Test $R^2$ |
| :--- | :--- | :--- |
| **Lasso Regression** | **0.7037** | **0.5282** |
| Stepwise OLS | 0.7187 | 0.5079 |
| Robust Regression | 0.7234 | 0.5015 |
| GAM (Non-Linear) | 0.7464 | 0.4693 |

**Conclusion:** The **Lasso Regression** model provided the best balance between bias and variance, outperforming complex non-parametric methods which suffered from overfitting due to the small sample size.

## 📦 Dependencies

To run the analysis code, you will need the following R packages:

```r
install.packages(c("ggplot2", "car", "lmtest", "MASS", "corrplot", "glmnet", "mgcv", "tidyr", "dplyr"))
