import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
import warnings


df = pd.read_csv("../../data/processed/T2_sample_per1_accelerometer.csv")

msk = np.random.rand(len(df)) < 0.8

train = df[msk]
test = df[~msk]

train_l = train[['labels']]
test_l = test[['labels']]
train = train.drop('labels', axis = 1)
test = test.drop('labels', axis = 1)

#Create a Gaussian Classifier
model = BernoulliNB()

# Train the model using the training sets 
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(train, train_l)

#Predict Output 
predicted= model.predict(test)

print("\n-------------------------\nUsing outliers:\n")

# Metrics
print("Metrics \n"  +classification_report(test_l, y_pred=predicted))

# Confussion matrix
print("Confussion Matrix:\n")
matrix = pd.crosstab(test_l['labels'], predicted, rownames=['actual'], colnames=['preds'])
print(matrix)


# 2nd approximation: Not using only outliers
#-------------------------------------------------------------

df = pd.read_csv("../../data/processed/T2_sample_per1_inliers_accelerometer.csv")

msk = np.random.rand(len(df)) < 0.8

train = df[msk]
test = df[~msk]

train_l = train[['labels']]
test_l = test[['labels']]
train = train.drop('labels', axis = 1)
test = test.drop('labels', axis = 1)

#Create a Gaussian Classifier
model = BernoulliNB()

# Train the model using the training sets 
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(train, train_l)

#Predict Output 
predicted= model.predict(test)

print("\n-------------------------\nUsing outliers + inliers:\n")

# Metrics
print("Metrics \n"  +classification_report(test_l, y_pred=predicted))

# Confussion matrix
print("Confussion Matrix:\n")
matrix = pd.crosstab(test_l['labels'], predicted, rownames=['actual'], colnames=['preds'])
print(matrix)

