import pandas as pd
from numpy import transpose
from sklearn import preprocessing 
from scipy import cluster
import sklearn.neighbors
import matplotlib.pyplot as plt
from pylab import savefig


df = pd.read_csv("../../data/interim/T2_Labels.csv")

# We now drop unimportant features, as Version, and every FFT or Middle_Sample column
df = df.drop(['UserID', 'UUID', 'Version', 'TimeStemp'], axis=1)

df = df[[c for c in df if "LinearAcceleration" in c or "AccelerometerStat" in c or "labels" in c]]
df = df[[c for c in df if "FFT" not in c and "MIDDLE_SAMPLE" not in c]]

# We proceed use hierarchical clustering.
scaler = preprocessing.MinMaxScaler()
datanorm = scaler.fit_transform(df)
dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(transpose(datanorm))
clusters = cluster.hierarchy.linkage(matsim, method = 'complete')
cluster.hierarchy.dendrogram(clusters, color_threshold=15)
savefig("../../reports/figures/Milestone3/Drendrogram_accelerometer")
plt.show()

# By looking at the dendrogram, we drop the most similar features
df = df.drop(df.columns[23], axis = 1)
df = df.drop(df.columns[22], axis = 1)
df = df.drop(df.columns[20], axis = 1)
df = df.drop(df.columns[16], axis = 1)
df = df.drop(df.columns[13], axis = 1)
df = df.drop(df.columns[5], axis = 1)

#---------------------------------------------------------------
# And repeeat the process
scaler = preprocessing.MinMaxScaler()
datanorm = scaler.fit_transform(df)
dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(transpose(datanorm))
clusters = cluster.hierarchy.linkage(matsim, method = 'complete')
cluster.hierarchy.dendrogram(clusters, color_threshold=20)
plt.show()


df = df.drop(df.columns[14], axis = 1)
df = df.drop(df.columns[1], axis = 1)


df.to_csv("../../data/processed/T2_Labels_Accelerometer.csv", index=False)



# ----------------------------------------------------------------
# Apply the same process to the samples

df = pd.read_csv("../../data/processed/T2_sample_per1.csv")

df = df[[c for c in df if "LinearAcceleration" in c or "AccelerometerStat" in c or "labels" in c]]
df = df[[c for c in df if "FFT" not in c and "MIDDLE_SAMPLE" not in c]]

df = df.drop(df.columns[23], axis = 1)
df = df.drop(df.columns[22], axis = 1)
df = df.drop(df.columns[20], axis = 1)
df = df.drop(df.columns[16], axis = 1)
df = df.drop(df.columns[13], axis = 1)
df = df.drop(df.columns[5], axis = 1)
df = df.drop(df.columns[14], axis = 1)
df = df.drop(df.columns[1], axis = 1)

df.to_csv("../../data/processed/T2_sample_per1_accelerometer.csv", index=False)


#------------------------------------------------------------------


df = pd.read_csv("../../data/processed/T2_sample_per1_inliers.csv")

df = df[[c for c in df if "LinearAcceleration" in c or "AccelerometerStat" in c or "labels" in c]]
df = df[[c for c in df if "FFT" not in c and "MIDDLE_SAMPLE" not in c]]

df = df.drop(df.columns[23], axis = 1)
df = df.drop(df.columns[22], axis = 1)
df = df.drop(df.columns[20], axis = 1)
df = df.drop(df.columns[16], axis = 1)
df = df.drop(df.columns[13], axis = 1)
df = df.drop(df.columns[5], axis = 1)
df = df.drop(df.columns[14], axis = 1)
df = df.drop(df.columns[1], axis = 1)

df.to_csv("../../data/processed/T2_sample_per1_inliers_accelerometer.csv", index=False)
