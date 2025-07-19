
# 📊 Customer Churn Prediction

This project aims to predict customer churn in a telecom company using various machine learning models. Churn prediction helps businesses understand why customers leave and how to retain them better.

---

## 📁 Dataset

**Source:** WA_Fn-UseC_-Telco-Customer-Churn.csv (Kaggle)

The dataset includes information such as:
- **Demographics:** gender, senior citizen, etc.
- **Services signed up:** InternetService, PhoneService, StreamingTV, etc.
- **Account info:** Contract type, tenure, payment method, MonthlyCharges, etc.
- **Target column:** `Churn` (Yes/No)

---

## 🔍 Data Preprocessing

- **Handling Missing Values:** Converted `TotalCharges` to numeric and removed blank records.
- **Encoding:** Label encoded all categorical variables using `LabelEncoder`.
- **Feature Scaling:** Used `StandardScaler` to scale numerical features.
- **Class Imbalance:** Balanced the dataset using SMOTE (Synthetic Minority Over-sampling Technique).

---

## 🧠 Model Training

The following models were used for training:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

### Tools Used:
```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
```

- Models were evaluated using classification metrics and confusion matrix.
- `GridSearchCV` was applied for hyperparameter tuning.

---

## ✅ Evaluation

- All models were compared using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
- Best performance was achieved using **XGBoost** and **Random Forest**.

---

## 💾 Model Saving

The final model was saved using `joblib`:

```python
import joblib
joblib.dump(model, "churn_model.pkl")
```

---

## 📈 Visualization

Data analysis and visualizations were done using:
- **Seaborn & Matplotlib** for heatmaps, countplots, histograms
- **Correlation matrix** for feature relationships


## 📌 Conclusion

This project demonstrates a complete ML pipeline for churn prediction:
- ✅ Data cleaning
- ✅ Feature engineering
- ✅ Model training
- ✅ Model evaluation
- ✅ Deployment-ready model

The insights can help telecom companies reduce churn and improve customer satisfaction.

---

## 👨‍💻 Author

**Swayam Kumar Gouda**  
📧 Email: swayamgouda2004@gmail.com  
🔗 GitHub: [skg2004](https://github.com/skg2004)  
🔗 LinkedIn: [swayam006](https://linkedin.com/in/swayam006)
