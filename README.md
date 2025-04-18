# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using various machine learning models. It includes implementations of Decision Tree, Logistic Regression, and Random Forest classifiers, along with a Flask web application for real-time prediction.​

## 📂 Repository Structure

Decision_tree.ipynb: Jupyter Notebook demonstrating the Decision Tree model training and evaluation.​

Decision_tree_flask_app.py: Flask application script for deploying the trained model.​

decision_tree_model.pkl: Serialized Decision Tree model for use in the Flask app.​

creditcardsampling.csv: Sample dataset used for model training and testing.​

finaldata.csv: Preprocessed dataset ready for model input.​

templates/: HTML templates for the Flask application's frontend.​

static/css/: CSS files for styling the Flask application's frontend.

## 📊 Dataset
The dataset used is a publicly available credit card transaction dataset, which includes both fraudulent and legitimate transactions. Due to confidentiality, feature names have been anonymized. The dataset is highly imbalanced, with fraudulent transactions constituting a small fraction of the total.​

## 🛠️ Features
Implementation of multiple machine learning models for fraud detection.​


Web application for real-time prediction using the trained model.​


Preprocessed datasets for immediate use.

## 📌 Note
The model's performance may vary due to the imbalanced nature of the dataset. It's recommended to explore techniques such as resampling or using different evaluation metrics (e.g., precision, recall, F1-score) for better assessment.​




