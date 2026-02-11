# 📊 Telco Customer Churn Prediction

## 🎯 Project Objective
This project predicts whether a telecom customer will churn (leave the service) using machine learning techniques. Customer churn prediction helps telecom companies retain customers and reduce revenue loss.

---

## 📂 Dataset
Dataset used: Telco Customer Churn Dataset (Kaggle)

The dataset contains customer demographic details, subscription information, billing data, and churn labels.

---

## ⚙️ Data Preprocessing
The following preprocessing steps were applied:

- Removed unnecessary column (customerID)
- Converted TotalCharges to numeric format
- Handled missing values using median imputation
- Encoded categorical variables using Label Encoding
- Scaled numerical features using StandardScaler

---

## 🔀 Train / Validation Split Method

The dataset was split using:

- 80% Training Data
- 20% Testing Data
  
The split was performed using Scikit-learn's `train_test_split` with:

---

## 🤖 Models Used

### 1️⃣ Baseline Model
- Logistic Regression
- Used to establish a simple reference performance

### 2️⃣ Improved Model
- Random Forest Classifier
- Captures nonlinear relationships and feature interactions

---

## 📏 Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

These metrics help evaluate classification performance and identify churn prediction errors.

---

## 📓 Project Notebook

The full workflow including data preprocessing, EDA, model training, and evaluation is available in Google Colab.

🔗 **Colab Link:**  
https://colab.research.google.com/drive/1Ah2a8VhgkT6aJLyhTJNtFsCrgsK5lOho#scrollTo=43pJ8L6Ry5RG

## 🏆 Best Result

| Model | Accuracy |
|---------|------------|
| Logistic Regression | ~80% |
| Random Forest | ~85% |

Random Forest achieved the best performance.

---

## 📊 Key Insights

- Customers with shorter tenure have higher churn probability.
- Customers with higher monthly charges tend to churn more.
- Contract type strongly influences churn.

---
