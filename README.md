# 🛡️ PhishShield – ML-based Phishing Website Detection

## 1. Problem Statement

Phishing websites are fake sites designed to steal sensitive information such as
passwords, banking details, and personal data. Traditional detection methods
(blacklists, rule-based filters, visual similarity checks) have limitations:
they often fail to catch new phishing techniques, require frequent manual
updates, or need high computing power.

**PhishShield** proposes a **machine learning–based phishing detection system**
that analyzes website URLs and automatically learns patterns of phishing versus
legitimate websites. By using classification algorithms, the system aims to:

- Improve accuracy in identifying phishing websites
- Reduce false alarms
- Provide near real-time protection
- Be more adaptive and reliable compared to static methods

---
## 2. Objectives

- Identify phishing websites using URL, domain features, and webpage properties
- Build a trained ML model for binary classification (Phishing / Legitimate)
- Deploy a user-friendly web interface for real-time predictions
- Improve cybersecurity awareness through intelligent detection

## 3. Key Features

✔ Extracts features automatically from the website URL
✔ Real-time prediction using trained ML model
✔ Simple and interactive web interface
✔ Lightweight & fast
✔ Can be further improved with more features (SSL, content, etc.)

## 4. Machine Learning Model

Algorithms Evaluated
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- Support Vector Machine (SVM)
- k-Nearest Neighbors (KNN)

After performance comparison, Random Forest was selected due to:
- High accuracy
- Robustness to noisy data

Ability to handle high-dimensional features
Model stored as:
📁 models/phishing_model.joblib

## 5. Installation & Setup

Requirements
- Install the required packages:
   pip install -r requirements.txt
- Run the Application
   cd webapp
   python app.py
- The app will run at:
    ➡ http://127.0.0.1:5000/

## 6. Model Training

- python src/train_model.py

Dataset should be placed inside:
data/raw/


## 7. Project Structure

```text
phishshield/
  data/
    raw/
      phishing_site_urls.csv      # Dataset (you place this file)
  models/
    phishshield_url_model.joblib  # Saved ML model after training
  src/
    __init__.py
    config.py
    data_utils.py
    train_model.py
    evaluate_model.py
    predict_url.py
  requirements.txt
  README.md
