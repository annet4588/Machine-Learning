import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# Read the dataset from a CSV file and store it in a pandas DataFrame
data = pd.read_csv("car.data")
# Print the first 5 rows of the dataset to understand its structure
print(data.head())

# Initialize the LabelEncoder to convert categorical data into numerical data
le = preprocessing.LabelEncoder()

# Encode the columns and store it in the related variables
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["safety"]))
safety = le.fit_transform(list(data["lug_boot"]))
cls = le.fit_transform(list(data["class"]))

# print(buying)

# Define the target variable (predictable variable)
predict = "class"

# Combine the feature columns into a list of tuples
X = list(zip(buying, maint, door, persons, lug_boot, safety))  # Features
# Define the labels (the target variable to be predicted)
y = list(cls)  # Labels

# Split the dataset into training and testing sets (90% training, 10% testing)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# print(x_train, y_test)

# Initialize the KNeighborsClassifier with 9 neighbors
model = KNeighborsClassifier(n_neighbors=9)  # Amount of Neighbors

# Train the KNeighborsClassifier model using the training data
model.fit(x_train, y_train)
# Calculate the accuracy of the model on the testing data
acc = model.score(x_test, y_test)
print(acc)

# Predict the labels for the testing data
predicted = model.predict(x_test)

# Define a list of class names corresponding to the numerical labels
names = ["unacc", "acc", "good", "vgood"]

# Loop through each prediction to compare it with the actual label
for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    # Find the k-nearest neighbors for the current test sample
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)