# Titanic Survival Prediction using Random Forest

## Project Overview
This repository focuses on predicting the survival of passengers aboard the Titanic using a **Random Forest Classifier**. The project leverages data preprocessing, feature engineering, and model tuning to achieve an accuracy of **82%** on test data, providing insights into the factors influencing survival.

## Features
- **Data Preprocessing:** Handling missing values and encoding categorical variables.
- **Exploratory Data Analysis (EDA):** Visualizing survival patterns based on passenger demographics and ticket class.
- **Feature Engineering:** Selecting the most impactful features for better model performance.
- **Model Implementation:** Developing and tuning a Random Forest Classifier.
- **Performance Evaluation:** Assessing model accuracy and visualizing results.

## Dataset
The dataset used for this project is the [Titanic Dataset](kaggle competitions download -c titanic) from Kaggle, which includes information on:
- Passenger demographics (Age, Gender, etc.)
- Socio-economic status (Pclass, Fare)
- Family details (SibSp, Parch)
- Survival outcome (1 = Survived, 0 = Not Survived)

## Project Steps

1. **Data Preprocessing:**
   - Imputed missing values for `Age` and `Embarked`.
   - Encoded categorical features like `Sex` and `Embarked` using one-hot encoding.

2. **Exploratory Data Analysis (EDA):**
   - Analyzed survival rates based on features such as `Pclass`, `Sex`, and `Age`.
   - Visualized correlations to identify key factors affecting survival.

3. **Feature Engineering:**
   - Selected significant features including `Pclass`, `Sex`, `Age`, `Fare`, `SibSp`, and `Parch`.

4. **Model Development:**
   - Implemented a **Random Forest Classifier** to predict survival.
   - Tuned hyperparameters such as `n_estimators`, `max_depth`, and `min_samples_split` to optimize performance.

5. **Model Evaluation:**
   - Evaluated model performance using accuracy, precision, recall, F1-score, and ROC-AUC curve.

## Results
The Random Forest model achieved an accuracy of **82%**, effectively predicting the survival of Titanic passengers.

## Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Repository Structure
```
Titanic-Survival-Prediction/
│
├── data/
│   └── titanic.csv
├── notebooks/
│   └── titanic_rf_model.ipynb
├── src/
│   ├── preprocess.py
│   ├── visualize.py
│   └── model.py
├── results/
│   └── evaluation_metrics.png
├── README.md
└── requirements.txt
```


## Future Work
- Implement other classification algorithms like XGBoost and SVM for comparison.
- Enhance feature engineering by including interaction terms and polynomial features.
- Deploy the model using a web framework like Flask or Streamlit.


Feel free to explore, fork, and contribute to the repository!

