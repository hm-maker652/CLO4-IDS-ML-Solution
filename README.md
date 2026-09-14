# AI-Powered Intrusion Detection System Using Machine Learning

## 1. Project Objective

The objective of this project is to develop an AI-powered Intrusion Detection System (IDS) that can identify normal and malicious network traffic using machine learning.

The project uses the CIC-IDS2017 dataset and applies data preprocessing, exploratory data analysis, and machine learning classification techniques. Three models were tested: Logistic Regression, Decision Tree, and Random Forest. A Streamlit dashboard was also developed to provide a user-friendly interface for exploring the data, viewing model performance, and making predictions.

## 2. Dataset Setup Instructions

This project uses the **CIC-IDS2017** dataset.

The dataset contains normal network traffic and different types of malicious network traffic, including attacks such as DoS, DDoS, Brute Force, Web Attacks, Infiltration, Botnet, and Heartbleed.

### Dataset Setup

1. Download the CIC-IDS2017 dataset.
2. Create a folder named `data` inside the project directory.
3. Place the required CSV dataset files inside the `data` folder.

The project expects the following structure:

```text
CLO4-IDS-ML-Solution/
│
├── IDS_Complete.ipynb
├── app.py
├── README.md
├── random_forest_model.pkl
├── feature_names.pkl
├── roc_data.pkl
├── normal_test.csv
├── malicious_test.csv
│
└── data/
    ├── dataset_file_1.csv
    ├── dataset_file_2.csv
    └── ...
```

The complete original dataset is not included in this repository because of its large size.

## 3. Data Preprocessing

The dataset was cleaned before training the models. The preprocessing steps included:

* Removing missing values.
* Removing duplicate records.
* Handling infinite values.
* Converting features into numeric format.
* Removing metadata columns such as IP addresses, Flow ID, and Timestamp.
* Creating a binary target variable:

  * `0` = Normal traffic
  * `1` = Malicious traffic
* Splitting the data into 80% training and 20% testing data.
* Applying feature scaling for Logistic Regression.
* Using class balancing to reduce the effect of class imbalance.

After preprocessing, the final dataset contained **743,697 records and 78 network traffic features**.

## 4. Machine Learning Models

The following machine learning models were trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Random Forest was selected as the final model because it achieved the highest ROC-AUC score and very high precision and recall.

## 5. How to Run the Code

### Requirements

Install Python and the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

### Run the Jupyter Notebook

Open the project folder and start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
IDS_Complete.ipynb
```

Run all cells from beginning to end.

### Run the Streamlit Dashboard

Open Command Prompt or Terminal in the project folder and run:

```bash
streamlit run app.py
```

The Streamlit dashboard will open in the browser.

The dashboard provides:

* Dashboard overview
* Data Explorer
* Model Training information
* Model Performance
* Network traffic prediction
* Feature Importance

## 6. Results Summary

Three machine learning models were evaluated using accuracy, precision, recall, F1-score, and ROC-AUC.

| Model               |   Accuracy |  Precision |     Recall |   F1-Score |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     93.02% |     72.73% |     97.22% |     83.21% |     98.39% |
| Decision Tree       |     99.81% |     99.48% |     99.43% |     99.46% |     99.66% |
| **Random Forest**   | **99.80%** | **99.56%** | **99.33%** | **99.44%** | **99.97%** |

Random Forest was selected as the final model because it achieved the highest ROC-AUC of **99.97%** and very high precision and recall.

The Random Forest confusion matrix contained:

* True Normal: 122,160
* False Positive: 116
* False Negative: 178
* True Malicious: 26,286

The results show that the proposed system can effectively distinguish between normal and malicious network traffic. However, false negatives remain an important security concern because malicious traffic incorrectly classified as normal may remain undetected.

## 7. Project Type

This project is a machine-learning-based proof-of-concept Intrusion Detection System. The Streamlit dashboard provides a prediction interface for network traffic samples. Future versions can be extended with live network traffic collection, automated alerts, and continuous monitoring.
