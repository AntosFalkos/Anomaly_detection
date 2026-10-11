# Anomaly Detection Pipeline

## Overview

An end-to-end Machine Learning pipeline designed to detect anomalies and potential fraud in high-volume system logs. This project tackles the common industry challenge of highly imbalanced tabular data. It utilizes advanced sampling techniques and compares two distinct modeling paradigms: strongly regularized tree-based models (XGBoost) and Multi-Layer Perceptrons (MLP).

## Data Privacy Note

Due to Non-Disclosure Agreements (NDA) and corporate intellectual property, the dataset provided in this repository is 100% synthetic. A dedicated script generates this mock data to perfectly mirror the statistical distributions, feature topology, and scale (reflecting the original \~5.9 million records) of the proprietary dataset. This allows for full code reproduction and logic verification without exposing any business secrets.

## Repository Structure

* `data/` - Contains the generated synthetic datasets (`df_controls.csv`, `df_visits.csv`, `synthetic_anomalies_data.csv`) for code reproduction.

* `images/` - Directory where output artifacts (Confusion Matrices, ROC curves, Feature Importances, Learning Curves) are saved after model evaluation.

* `notebooks/` - Jupyter Notebooks containing Exploratory Data Analysis (EDA) and results visualization.

* `src/` - Modularized Python source code:

  * `main.py` - Pipeline orchestrator executing the end-to-end workflow.

  * `preprocess.py` - Feature engineering, scaling, and class imbalance handling.

  * `train_xgb.py` - XGBoost model training pipeline and decision threshold optimization.

  * `train_mlp.py` - Neural Network training pipeline.

* `generate_mock_data.py` - Script to generate synthetic facility and visitor logs for pipeline execution.

* `requirements.txt` - Python environment dependencies and libraries.

## Methodology & Pipeline

### 1. Preprocessing and Class Imbalance Handling

Anomalies inherently represent a microscopic fraction of total system traffic. To handle this extreme imbalance without introducing Data Leakage:

* **Stratified Split:** Data was partitioned using a stratified split to maintain original class proportions in the validation/test sets.

* **Oversampling:** The SMOTE (Synthetic Minority Over-sampling Technique) algorithm was applied strictly to the training set to prevent the model from favoring the majority class.

* **Scaling:** StandardScaler was used to normalize feature distributions, a critical step for stable convergence and weight optimization in the Neural Network architecture.

### 2. Model Architectures

Instead of relying on a single algorithm, this project contrasts two distinct approaches to evaluate the Precision-Recall trade-off.

**A. Extreme Gradient Boosting (XGBoost)**

* **Design:** Optimized for maximum anomaly capture in tabular data.

* **Regularization:** Since SMOTE can lead to overfitting, the XGBoost model was heavily regularized using hyperparameters such as shallow trees (`max_depth=3`), high minimum child weight (`min_child_weight=13`), and strict penalties (`reg_lambda=12.0`, `gamma=2.0`).

* **Tuning:** The decision threshold was manually optimized for the AUCPR (Area Under the Precision-Recall Curve) metric to ensure a minimum Recall of >= 90%, intentionally disregarding the misleading Accuracy metric in this context.

**B. Multi-Layer Perceptron (MLP)**

* **Design:** A Fully Connected neural network designed for tabular data.

* **Architecture:** Two hidden layers (64 and 32 neurons) utilizing the ReLU activation function, outputting to a Sigmoid layer for probability mapping.

* **Optimization:** Trained using the Adam optimizer with Log-Loss over 100 epochs. Learning curves demonstrated error stabilization around epoch 20-25.

## Business Insights (Trade-Off)

The key takeaway from this project is the illustration of the classic Precision-Recall trade-off, opening the door to two different deployment strategies depending on organizational needs:

* **The XGBoost Approach (High Recall / Aggressive):** Through decision threshold optimization, this model captures almost all anomalies (Recall > 90%) but generates a higher volume of False Positives.

  * *Business Application:* Best suited for environments with a dedicated Security/Fraud Operations team capable of manually verifying flagged, suspicious events.

* **The Neural Network Approach (High Precision / Conservative):** While capturing a lower total number of anomalies, the MLP model demonstrates a superior ability to distinguish between classes, leading to a drastic reduction in False Positives.

  * *Business Application:* Ideal for fully automated systems (e.g., auto-blocking a suspicious user session or API request). It minimizes the risk of blocking legitimate clients, thereby protecting User Experience (UX) and system continuity.

## Quick Start (Reproduction)

Clone the repository:

```bash
git clone https://github.com/AntosFalkos/Anomaly_detection.git
cd Anomaly_detection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Generate the synthetic dataset (Data Generation):

```bash
python generate_mock_data.py
```

Run the full pipeline (Preprocessing -> Training -> Evaluation):

```bash
python src/main.py
```