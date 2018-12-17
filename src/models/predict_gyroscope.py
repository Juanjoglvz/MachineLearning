import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

import warnings

# Taking important features

df = pd.read_csv("../data/processed/T2_sample_per1_gyroscope.csv")


# Split into test and train datasets

msk = np.random.rand(len(df)) < 0.8

train = df[msk]
test = df[~msk]

train_l = train[['labels']]
test_l = test[['labels']]
train = train.drop('labels', axis = 1)
test = test.drop('labels', axis = 1)

#Create a Random Forest model
model = GaussianNB()

# Train the model using the training sets 
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(train, train_l)

#Predict Output 
predicted= model.predict(test)

print("\n-------------------------\nUsing outliers with Naive-Bayes:\n")

# Metrics
print("Metrics \n"  +classification_report(test_l, y_pred=predicted))

# Confussion matrix
print("Confussion Matrix:\n")
matrix = pd.crosstab(test_l['labels'], predicted, rownames=['actual'], colnames=['preds'])
print(matrix)


# 2nd approximation: Not using only outliers
#-------------------------------------------------------------
# Taking important features

df = pd.read_csv("../data/processed/T2_sample_per1_inliers_gyroscope.csv")

df = df[[col for col in df.columns if ("Gyroscope" in col and "FFT" not in col) or "labels" == col]]

# Split into test and train datasets

msk = np.random.rand(len(df)) < 0.8

train = df[msk]
test = df[~msk]

train_l = train[['labels']]
test_l = test[['labels']]
train = train.drop('labels', axis = 1)
test = test.drop('labels', axis = 1)

#Create a Random Forest model
model = RandomForestClassifier(n_estimators=2000, random_state=150)

# Train the model using the training sets 
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(train, train_l)

#Predict Output 
predicted= model.predict(test)

print("\n-------------------------\nUsing outliers + inliers with Random Forest:\n")

# Metrics
print("Metrics \n"  +classification_report(test_l, y_pred=predicted))

# Confussion matrix
print("Confussion Matrix:\n")
matrix = pd.crosstab(test_l['labels'], predicted, rownames=['actual'], colnames=['preds'])
print(matrix)