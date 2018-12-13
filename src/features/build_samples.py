from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import numpy as np
from sklearn import preprocessing


df = pd.read_csv("../../data/interim/T2_Labels.csv")
df = df.drop(['UserID', 'UUID', 'Version', 'TimeStemp'], axis=1)

####### OUTLIERS : PERCENTILE 1
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

outliers.to_csv("../../data/processed/T2_sample_per1.csv", index = False)


####### OUTLIERS + INLIERS : PERCENTILE 1
percentile = 0.5
n_outliers = len(X_scores[X_scores < np.percentile(X_scores, percentile)])
lowest_indices = np.argsort(-X_scores, axis=0)[-1:-1-n_outliers:-1]
outliers = df.iloc[lowest_indices]

attacks = df[(df['labels'] == 1)]
outliers = outliers[(outliers['labels'] != 1)]
outliers = outliers.append(attacks)

inliers = df.sample(n_outliers)
inliers = inliers[(inliers['labels'] == 0)]
inliers = inliers[[i for i in inliers if i.index not in lowest_indices]]

merged = outliers.append(inliers)

merged.to_csv("../../data/processed/T2_sample_per1_inliers.csv", index = False)
