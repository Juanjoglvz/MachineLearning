import pandas as pd
from numpy import transpose
from sklearn import preprocessing 
from scipy import cluster
import sklearn.neighbors

df = pd.read_csv("../../data/interim/T2_Labels.csv")

# We now drop unimportant features, as Version, and every FFT or Middle_Sample column
df = df.drop(['UserID', 'UUID', 'Version', 'TimeStemp'], axis=1)
df = df[[c for c in df if "FFT" not in c and "MIDDLE_SAMPLE" not in c]]


# We assume that what we learnt for first/second day applies to the whole dataset
# See build_features_Accelerometer_Day2 and build_features_Gyroscope from Milestone 1
df = df.drop("AccelerometerStat_x_MEDIAN", axis = 1)
df = df.drop("AccelerometerStat_y_MEDIAN", axis = 1)
df = df.drop("AccelerometerStat_z_MEDIAN", axis = 1)
df = df.drop("LinearAcceleration_x_MEDIAN", axis = 1)
df = df.drop("LinearAcceleration_y_MEDIAN", axis = 1)
df = df.drop("LinearAcceleration_z_MEDIAN", axis = 1)
df = df.drop("AccelerometerStat_x_VAR", axis = 1)
df = df.drop("AccelerometerStat_y_VAR", axis = 1)
df = df.drop("AccelerometerStat_z_VAR", axis = 1)

df = df.drop("GyroscopeStat_y_MEDIAN", axis = 1)
df = df.drop("GyroscopeStat_z_MEDIAN", axis = 1)
df = df.drop("RotationVector_xSinThetaOver2_MEDIAN", axis = 1)
df = df.drop("RotationVector_ySinThetaOver2_MEAN", axis = 1)
df = df.drop("RotationVector_ySinThetaOver2_MEDIAN", axis = 1)
df = df.drop("RotationVector_zSinThetaOver2_MEDIAN", axis = 1)
df = df.drop("GyroscopeStat_y_VAR", axis = 1)
df = df.drop("GyroscopeStat_z_VAR", axis = 1)

# We proceed use hierarchical clustering.
scaler = preprocessing.MinMaxScaler()
datanorm = scaler.fit_transform(df)
dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(transpose(datanorm))
clusters = cluster.hierarchy.linkage(matsim, method = 'complete')

# Instead of looking at the dendrogram, wich is going to be difficult to interpret
# as we have 47 features, we'll use the array "clusters"

# After checkin what columns they are, we'll drop features number 27, 22 and 15
# by looking at the third column of clusters
print(df.columns[27])
print(df.columns[22])
print(df.columns[15])

df = df.drop(df.columns[27], axis = 1)
df = df.drop(df.columns[22], axis = 1)
df = df.drop(df.columns[15], axis = 1)

# -----------------------------------------------------
# Now we repeat the process again

scaler = preprocessing.MinMaxScaler()
datanorm = scaler.fit_transform(df)
dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(transpose(datanorm))
clusters = cluster.hierarchy.linkage(matsim, method = 'complete')

df = df.drop(df.columns[31], axis = 1)
df = df.drop(df.columns[20], axis = 1)
df = df.drop(df.columns[18], axis = 1)

#------------------------------------------------------
# And repeat again

scaler = preprocessing.MinMaxScaler()
datanorm = scaler.fit_transform(df)
dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(transpose(datanorm))
clusters = cluster.hierarchy.linkage(matsim, method = 'complete')

df = df.drop(df.columns[39], axis = 1)
df = df.drop(df.columns[37], axis = 1)
df = df.drop(df.columns[32], axis = 1)
df = df.drop(df.columns[27], axis = 1)
df = df.drop(df.columns[17], axis = 1)
df = df.drop(df.columns[13], axis = 1)
df = df.drop(df.columns[10], axis = 1)

# There is a good indicator for stopping: in the second row we can see
# feature 33, wich is our label for MOriarty attack

df.to_csv("../../data/processed/T2_Labels_Processed.csv", index=False)