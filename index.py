
#Diabetes Prediction Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

import xgboost as xgb
import pickle


#Loading Dataset

df = pd.read_csv("diabetes.csv")  #CSV file (same folder)
print("Dataset shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())
print("\nOutcome counts:\n", df['Outcome'].value_counts())

#Exploratory Data Analysis
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

sns.countplot(x='Outcome', data=df)
plt.title("Diabetes Outcome Count")
plt.show()

#Histograms
df.hist(figsize=(12,10))
plt.tight_layout()
plt.show()

#Data Preprocessing
#Replacing 0s with NaN in certain columns
cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_zeros] = df[cols_with_zeros].replace(0, pd.NA)

#Fill missing values with median
df.fillna(df.median(), inplace=True)

#Features and Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Model Training

#Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

#Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

#XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))

#Model Evaluation
#ROC Curve & AUC for Logistic Regression
y_proba_lr = lr.predict_proba(X_test)[:,1]
roc_auc_lr = roc_auc_score(y_test, y_proba_lr)
print("Logistic Regression ROC-AUC:", roc_auc_lr)

fpr, tpr, thresholds = roc_curve(y_test, y_proba_lr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_lr:.2f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.show()

#Feature Importance (Random Forest)
importances = rf.feature_importances_
feat_names = X.columns

plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=feat_names)
plt.title("Feature Importance (Random Forest)")
plt.show()

#Optional: Hyperparameter Tuning (Random Forest)
param_grid = {
    'n_estimators': [50,100,200],
    'max_depth': [None,5,10],
    'min_samples_split': [2,5,10]
}

grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print("Best Random Forest Params:", grid.best_params_)

#Save Model for Deployment
#Save Logistic Regression model
pickle.dump(lr, open("diabetes_model.pkl","wb"))
print("Model saved as 'diabetes_model.pkl'")
