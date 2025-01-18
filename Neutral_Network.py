import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier

from sklearn import datasets
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sb

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

# Create training/ test data split
# Split the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Neutral Network Classification model
nn_model = MLPClassifier(hidden_layer_sizes=(10,100), activation='relu', solver='adam', max_iter=300, random_state=42)

# Train the model
nn_model.fit(x_train, y_train)
y_pred = nn_model.predict(x_test)


print("Y Prediction:")
print (y_pred)

# Calculate Accuracy
nn_classifier_accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {nn_classifier_accuracy}")

# Report
print("\nSurvival Classification Report:")
print(classification_report(y_test, y_pred))

# Calculate the stats to compare
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print out results
print(f"Mean squared error: {mse}")
print(f"Mean absolute error: {mae}")
print(f"Root mean squared error: {rmse}")
print(f"R2 score: {r2}")


# Visualize with a confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
sb.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Dead', 'Pred Survived'], yticklabels=['Act Dead', 'Act Survived'])

# Add labels and title
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Titanic Survival Confusion Matrix Heatmap')
plt.show()
