import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

# Load the data
data = pd.read_csv("student-mat.csv", sep=";")
print(data.head())

# Select relevant columns
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# data = shuffle(data)  # Optional - shuffle the data
print(data.head())

# Define the target variable
predict = "G3"

X = np.array(data.drop([predict], 1)) # Features (all columns except G3)
y = np.array(data[predict]) # Labels / Target (G3 column)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
"""best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))

    if acc > best:
        best = acc
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)"""

# LOAD MODEL
pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)


print("-------------------------")
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print("-------------------------")

predicted= linear.predict(x_test)
for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x])


# Drawing and plotting model
plot = "G2"
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()

#
# linear = linear_model.LinearRegression()
#
# linear.fit(x_train, y_train)
# acc = linear.score(x_test, y_test) # acc stands for accuracy
#
# print(f'Accuracy: \t', acc)
#
# print('Coefficient: \n', linear.coef_)  # These are each slope value
# print('Intercept: \n', linear.intercept_)  # This is the intercept
#
# predictions = linear.predict(x_test) # Gets a list of all predictions
#
# for x in range(len(predictions)):
#     print(predictions[x], x_test[x], y_test[x])

