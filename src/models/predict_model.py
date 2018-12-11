from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import BernoulliNB

df = pd.read_csv("../../data/processed/T2_Labels_Processed.csv")

clf = LocalOutlierFactor(n_neighbors=4, contamination=0.1)

scaler = preprocessing.MinMaxScaler()
datanorm = scaler.fit_transform(df)
y_pred = clf.fit_predict(datanorm)

X_scores = clf.negative_outlier_factor_

percentile = 1
n_outliers = len(X_scores[X_scores < np.percentile(X_scores, percentile)])
lowest_indices = np.argsort(-X_scores, axis=0)[-1:-1-n_outliers:-1]
outliers = df.iloc[lowest_indices]

attacks = df[(df['labels'] == 1)]

outliers = outliers[(outliers['labels'] != 1)]

outliers = outliers.append(attacks)

#-----------------------------
msk = np.random.rand(len(outliers)) < 0.8

train = outliers[msk]
test = outliers[~msk]

train_l = train[['labels']]
test_l = test[['labels']]
train = train.drop('labels', axis = 1)
test = test.drop('labels', axis = 1)

#Create a Gaussian Classifier
model = BernoulliNB()

# Train the model using the training sets 
model.fit(train, train_l)

#Predict Output 
predicted= model.predict(test)

print(test_l.iloc[0])


# Metrics
tn = 0
fn = 0
tp = 0
fp = 0
i = 0
for index, row in test_l.iterrows():
    if predicted[i] == 0 and row['labels'] == 0:
        tn = tn +1
    elif predicted[i] == 0 and row['labels'] == 1:
        fn = fn +1
    elif predicted[i] == 1 and row['labels'] == 1:
        tp = tp +1
    else:
        fp = fp +1
    i = i+1

print (str(tn)+" True negatives, "+str(fn)+ " False negatives")
print (str(tp)+ " True positives, "+str(fp)+ " False positives")