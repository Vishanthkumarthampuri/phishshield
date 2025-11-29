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

## 2. Project Structure

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
