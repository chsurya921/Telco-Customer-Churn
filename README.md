# Telco-Customer-Churn
Objective
The goal of this project is to predict whether a telecom customer will churn (leave the service) using machine learning techniques. This helps telecom companies identify at-risk customers and take proactive retention actions.
Dataset
Dataset Used: Telco Customer Churn Dataset (Kaggle)

The dataset contains customer demographic details, subscription information, billing data, and churn labels (Yes/No).
Data Preprocessing
- Removed unnecessary column: customerID
- Converted TotalCharges to numeric format
- Handled missing values using median imputation
- Encoded categorical variables using Label Encoding
- Scaled numerical features using StandardScaler
Train / Validation Split Method
The dataset was split using an 80-20 ratio:
- 80% Training Data
- 20% Testing Data

The split was performed using train_test_split with test_size=0.2 and random_state=42 to ensure reproducibility.
Models Used
1. Logistic Regression (Baseline Model)
2. Random Forest Classifier (Improved Model)
Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
Final Results
Logistic Regression:
Accuracy: 80%
Precision: 0.68
Recall: 0.56
F1 Score: 0.61

Random Forest (Best Model):
Accuracy: 85%
Precision: 0.76
Recall: 0.65
F1 Score: 0.70

The Random Forest model achieved the best overall performance.
Key Insights
- Customers with short tenure are more likely to churn.
- Customers with higher monthly charges show higher churn probability.
- Month-to-month contract customers have higher churn rates.
<img width="468" height="639" alt="image" src="https://github.com/user-attachments/assets/a741a793-e2f2-4b82-a7b3-6e5b95500e01" />
