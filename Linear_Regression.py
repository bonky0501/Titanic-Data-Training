import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load data from CSV
data = pd.read_csv('titanic_training_data.csv')

# Select the columns we are going to analyze
columnsFromDataset = ['Survived', 'Sex', 'Age', 'Pclass', 'SibSp', 'Parch']
data = data[columnsFromDataset]

# Prepare the columns that have missing or incompatible data
data['Sex'] = data['Sex'].map({'male': 1, 'female': 0}) # Maps string values to numbers
data = data.dropna(subset = ['Age']) # Drop ROWS that have missing data

# These are the features that we will be analyzing
columnsForFeatures = ['Sex', 'Age', 'Pclass', 'SibSp', 'Parch']

# Setup the x and y columns
featuresX = data[columnsForFeatures]
targetY = data['Survived']

# Split the data into training and testing
featuresX_train, featuresX_test, targetY_train, targetY_test = train_test_split(featuresX, targetY, test_size=0.2, random_state=42)

# Linear regression
lin_model = LinearRegression()
lin_model.fit(featuresX_train, targetY_train)
targetY_prediction = lin_model.predict(featuresX_test)

# Calculate error
mae = mean_absolute_error(targetY_test, targetY_prediction)
mse = mean_squared_error(targetY_test, targetY_prediction)
rmse = np.sqrt(mean_squared_error(targetY_test, targetY_prediction))
r2 = r2_score(targetY_test, targetY_prediction)

# Print result
print(f"Mean absolute error: {mae}")
print(f"Mean squared error: {mse}")
print(f"Root mean squared error: {rmse}")
print(f"R2 score: {r2}")

# Visualize the regression line
# This graph is specifically showing Sex on the X axis
# It is a mess because linear regression is a poor fit for this data and for using it to determine if someone survived or not
plt.scatter(featuresX_test.iloc[:, 0], targetY_test, color='blue')
plt.plot(featuresX_test.iloc[:, 0], targetY_prediction, color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.show()

