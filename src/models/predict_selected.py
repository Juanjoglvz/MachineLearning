import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import warnings

# Taking important features

df = pd.read_csv("../../data/processed/T2_sample_per1.csv")

df = df[['AccelerometerStat_x_MEAN', 'MagneticField_z_MEAN', 'AccelerometerStat_z_MEAN', 'MagneticField_y_MEAN', 'LinearAcceleration_COV_z_y', 'AccelerometerStat_y_MEAN', 'LinearAcceleration_z_MEAN', 'labels']]

# Split into test and train datasets

msk = np.random.rand(len(df)) < 0.8

train = df[msk]
test = df[~msk]

train_l = train[['labels']]
test_l = test[['labels']]
train = train.drop('labels', axis = 1)
test = test.drop('labels', axis = 1)

#Create a Random Forest model
model = RandomForestClassifier(n_estimators=1000, random_state=7)

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
# Taking important features

df = pd.read_csv("../../data/processed/T2_sample_per1_inliers.csv")

df = df[['AccelerometerStat_x_MEAN', 'MagneticField_z_MEAN', 'AccelerometerStat_z_MEAN', 'MagneticField_y_MEAN', 'LinearAcceleration_COV_z_y', 'AccelerometerStat_y_MEAN', 'LinearAcceleration_z_MEAN', 'labels']]

# Split into test and train datasets

msk = np.random.rand(len(df)) < 0.8

train = df[msk]
test = df[~msk]

train_l = train[['labels']]
test_l = test[['labels']]
train = train.drop('labels', axis = 1)
test = test.drop('labels', axis = 1)

#Create a Random Forest model
model = RandomForestClassifier(n_estimators=1000, random_state=7)

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