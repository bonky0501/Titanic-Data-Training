import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold,cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


# Load data from CSV
data = pd.read_csv('titanic_training_data.csv')

# Select the columns we are going to analyze
columnsFromDataset = ['Survived', 'Sex', 'Age', 'Pclass', 'SibSp', 'Parch']
data = data[columnsFromDataset]

# Prepare the columns that have missing or incompatible data
data['Sex'] = data['Sex'].map({'male': 1, 'female': 0}) # Maps string values to numbers
data = data.dropna(subset = ['Age']) 

# These are the features that we will be analyzing
columnsForFeatures = ['Sex', 'Age', 'Pclass', 'SibSp', 'Parch']

# Setup the x and y columns
x = pd.DataFrame(data[columnsForFeatures].values, columns = columnsForFeatures)
y = pd.Series(data['Survived'])

# Evaluate cross_validation for Linear Regression
lin_model = LinearRegression()
lin_score = cross_val_score(lin_model, x, y, cv = 5)
print("_______Linear Regression_______")
print("Cross Validation Scores: ", lin_score)
print("Average CV Score: ", lin_score.mean())  
print("Number of CV Scores used in Average: ", len(lin_score))
print("\n")

# Evaluate cross_validation for Logistic Regression
log_model = LogisticRegression(max_iter=200)
log_score = cross_val_score(log_model, x, y, cv = 5)
print("_______Logistic Regression_______")
print("Cross Validation Scores: ", log_score)
print("Average CV Score: ", log_score.mean())  
print("Number of CV Scores used in Average: ", len(log_score))
print("\n")


# Evaluate cross_validation for Support Vecter Machine
svm_model = SVC(kernel='linear')
svm_score = cross_val_score(svm_model, x, y, cv = 5)
print("_______Support Vecter Machine_______")
print("Cross Validation Scores: ", svm_score)
print("Average CV Score: ", svm_score.mean())  
print("Number of CV Scores used in Average: ", len(svm_score))
print("\n")

# Evaluate cross_validation for Neutral Network
nn_model = MLPClassifier(hidden_layer_sizes=(10,100), activation='relu', solver='adam', max_iter=500, random_state=42)
nn_score = cross_val_score(nn_model, x, y, cv = 5)
print("_______Neutral Network_______")
print("Cross Validation Scores: ", nn_score)
print("Average CV Score: ", nn_score.mean())  
print("Number of CV Scores used in Average: ", len(nn_score))
print("\n")

# Evaluate cross_validation for Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_score = cross_val_score(rf_model, x, y, cv = 5)
print("_______Random Forest_______")
print("Cross Validation Scores: ", rf_score)
print("Average CV Score: ", rf_score.mean())  
print("Number of CV Scores used in Average: ", len(rf_score))
print("\n")



