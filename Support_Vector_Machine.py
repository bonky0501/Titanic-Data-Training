import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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
data = data.dropna(subset = ['Age']) # Drop ROWS that have missing data

# These are the features that we will be analyzing
columnsForFeatures = ['Sex', 'Age', 'Pclass', 'SibSp', 'Parch']

# Setup the x and y columns
featuresX = data[columnsForFeatures]
targetY = data['Survived']

# Split the data into training and testing
featuresX_train, featuresX_test, targetY_train, targetY_test = train_test_split(featuresX, targetY, test_size=0.2, random_state=42)

# Support Vector Machine
svm_model = SVC(kernel='linear')
svm_model.fit(featuresX_train, targetY_train)
targetY_prediction = svm_model.predict(featuresX_test)

### Full model report
# Accuracy
svm_model_accuracy = accuracy_score(targetY_test, targetY_prediction)
print(f"Model Accuracy: {svm_model_accuracy}")

# Classification
print("\nSurvival Classification Report:")
print(classification_report(targetY_test, targetY_prediction))

# Calculate error stats to compare to other models
mae = mean_absolute_error(targetY_test, targetY_prediction)
mse = mean_squared_error(targetY_test, targetY_prediction)
rmse = np.sqrt(mean_squared_error(targetY_test, targetY_prediction))
r2 = r2_score(targetY_test, targetY_prediction)

# Print error results
# Clearly these are not a good way of judging the aptitude of the model
print(f"Mean absolute error: {mae}")
print(f"Mean squared error: {mse}")
print(f"Root mean squared error: {rmse}")
print(f"R2 score: {r2}")

# Visualize with a confusion matrix
confusion_matrix = confusion_matrix(targetY_test, targetY_prediction)
sb.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Dead', 'Pred Survived'], yticklabels=['Act Dead', 'Act Survived'])

# Add labels and title
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Titanic Survival Confusion Matrix Heatmap')
plt.show()