## Diabetes Prediction using Machine Learning

This project predicts whether a person is diabetic based on medical diagnostic measurements. It uses machine learning models to classify patients as diabetic or non-diabetic based on features such as glucose level, BMI, insulin level and more.


#Project Overview

The goal of this project is to build and evaluate machine learning models that can accurately predict diabetes using the PIMA Indians Diabetes dataset. The project includes data preprocessing, exploratory analysis, model training, evaluation, and deployment of a prediction system.


#Features

Data cleaning and preprocessing

Exploratory data analysis

Feature scaling and model building

Comparison of multiple ML algorithms

Trained model saved for deployment

Streamlit web app for user-friendly predictions


#Technologies Used

Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-Learn
Streamlit
Pickle


#Project Structure

Diabetes-Prediction/
data/
diabetes.csv
models/
diabetes_model.pkl
app.py
model_training.py
requirements.txt
README.md
.gitignore


#How to Run the Project

Clone the repository
git clone https://github.com/sarveshgedam/Diabetes-Prediction.git

cd Diabetes-Prediction

Install dependencies
pip install -r requirements.txt

Train the model (optional)
python model_training.py

Run the Streamlit App
streamlit run app.py
Open the URL shown in the terminal to use the prediction interface.


#Model Details

The project evaluates multiple machine learning models including Logistic Regression, Support Vector Machine, Random Forest Classifier, and K-Nearest Neighbors.
The final model is selected based on accuracy, precision, recall, and F1-score.


#Dataset

The dataset used is the PIMA Indians Diabetes Dataset which contains 768 samples and 8 medical predictor variables.


#Contributing

Contributions are welcome. Fork the repository and submit a pull request for improvements.
