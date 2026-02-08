# Credit Risk Prediction with XGBoost, Continuous Retraining and MLflow Tracking

## 1. Abstract

Credit risk evaluation is a fundamental task in financial decision-making because incorrect credit approvals may generate significant economic losses. This project addresses the problem of predicting whether a client will repay or not repay a credit obligation using supervised machine learning. The proposed solution is an end-to-end credit risk prediction system based on XGBoost, integrated with a continuous learning pipeline and MLflow experiment tracking.

The dataset used is structured German-credit style tabular data containing demographic and financial variables such as age, credit amount, duration, employment status, savings, and checking account condition. Model quality is evaluated using Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix, ROC curve, Precision-Recall curve, and probability distribution analysis.

Results show stable and reproducible predictive performance across retraining cycles. As future work, the system can be extended with class-imbalance handling, threshold optimization, and ensemble modeling approaches.

---

## Technologies and Tools (Stack)

- **Machine Learning & Data Science:** Python, Pandas, NumPy, Scikit-learn, XGBoost  
- **Experiment Tracking:** MLflow (runs, metrics, artifacts, model versioning)  
- **Backend:** FastAPI, Uvicorn  
- **Visualization:** Matplotlib (confusion matrix, ROC, PR curve, probability histograms)

---

## System Interface

The application provides a web interface to evaluate credit applications and later register real payment outcomes used for retraining.

![System UI](german-credit-continuous-training-mlflow/artifacts/figures/app_fuction.png)

---

## 2. Proposed Method

The proposed method follows a structured machine learning lifecycle combined with a deployment and feedback loop for continuous retraining.

The main stages are:

- Exploratory Data Analysis (EDA)
- Data preprocessing and transformation
- Feature encoding and scaling
- Supervised training with XGBoost
- Model evaluation with multiple metrics
- Experiment tracking with MLflow
- Deployment through FastAPI
- User feedback capture
- Incremental retraining with new labeled data
- Model version replacement

The preprocessing stage uses Scikit-learn pipelines including missing value imputation, numerical scaling, and One-Hot encoding for categorical variables. The classifier is an XGBoost binary model that outputs repayment probability.

Each training and retraining run logs parameters, metrics, curves, and model artifacts into MLflow for reproducibility.

### Proposed Method Diagram

![Proposed Method Diagram](german-credit-continuous-training-mlflow/artifacts/figures/diagrama_ui.png)

### Table 1 — Model Parameters

| Parameter | Value |
|------------|------------|
| Algorithm | XGBoost Classifier |
| n_estimators | 200 |
| learning_rate | 0.1 |
| max_depth | 5 |
| eval_metric | logloss |
| random_state | 42 |

---

## 3. Experimental Design

### 3.1 Dataset Characteristics

The dataset is structured tabular credit data with variables including:

- Credit duration
- Credit amount
- Age
- Checking account status
- Employment status
- Savings category
- Credit purpose

Target variable:

- 1 = Paid  
- 0 = Not Paid

The dataset is appropriate for supervised binary classification using gradient boosting algorithms.

---

### 3.2 Training Configuration

The model uses gradient boosting trees (XGBoost) due to strong performance on structured financial datasets. Fixed hyperparameters are used for reproducibility and fair comparison across retraining runs. Each experiment logs metrics and artifacts through MLflow.

---

## 4. Results and Discussion

Following the proposed method and retraining workflow, the system generates evaluation artifacts automatically stored in MLflow.

### Evaluation Metrics (Latest Retraining Run)

| Metric | Value |
|---------|---------|
| Accuracy | **0.70** |
| Precision | **0.74** |
| Recall | **0.87** |
| F1-score | **0.80** |
| ROC-AUC | **0.71** |

These results indicate good recall and solid F1 performance. In credit risk evaluation, higher recall is desirable to reduce undetected risky clients.

---

### Confusion Matrix

![Confusion Matrix](german-credit-continuous-training-mlflow/artifacts/figures/v_20260204_204053_confusion_matrix.png)

The confusion matrix shows strong detection of paid cases with moderate false positives, suggesting that threshold calibration could further improve balance.

---

### Precision–Recall Curve

![PR Curve](german-credit-continuous-training-mlflow/artifacts/figures/v_20260204_204053_pr_curve.png)

The Precision–Recall curve indicates stable precision at medium recall ranges and expected degradation near full recall.

---

### ROC Curve

![ROC Curve](german-credit-continuous-training-mlflow/artifacts/figures/v_20260204_204053_roc_curve.png)

ROC-AUC ≈ **0.71**, confirming performance above random baseline.

---

### Probability Distribution

![Probability Histogram](german-credit-continuous-training-mlflow/artifacts/figures/v_20260204_204053_proba_hist.png)

Probability distributions show class separation behavior and support threshold tuning strategies.

---

## 5. Conclusions

This work presents a complete and reproducible machine learning system for credit risk prediction using XGBoost with continuous retraining and MLflow tracking. The methodology integrates EDA, preprocessing pipelines, supervised learning, experiment tracking, real-time inference, feedback collection, and incremental retraining.

Results confirm that gradient boosting models are effective for structured credit data and that continuous retraining pipelines are feasible in practical decision-support systems. MLflow tracking guarantees reproducibility, model version control, and experiment comparability.

Future work includes class imbalance handling, threshold optimization, ensemble models, and expanded financial feature sets.

---

## References

- UCI Machine Learning Repository — German Credit Dataset  
  https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

- XGBoost Documentation  
  https://xgboost.readthedocs.io


---

## Authors

Juan Guillen  
juanito.albertog6@gmail.com  

Ariel Solano  
arisolri1@gmail.com
