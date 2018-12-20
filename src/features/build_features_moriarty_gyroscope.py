import pandas as pd
from numpy import transpose
from sklearn import preprocessing 
from scipy import cluster
import sklearn.neighbors
import matplotlib.pyplot as plt
from pylab import savefig


df = pd.read_csv("../data/interim/T2_Labels.csv")

# Get only objective data
df = df[[col for col in df.columns if ("Gyroscope" in col and "FFT" not in col) or "labels" == col]]

# We proceed use hierarchical clustering.
scaler = preprocessing.MinMaxScaler()
datanorm = scaler.fit_transform(df)
dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(transpose(datanorm))
clusters = cluster.hierarchy.linkage(matsim, method = 'complete')
cluster.hierarchy.dendrogram(clusters, color_threshold=15)
savefig("../../reports/figures/Milestone3/Dendrogram_gyroscope")
plt.show()

# By looking at the dendrogram, we drop the most similar features
df = df.drop(df.columns[12], axis = 1)
df = df.drop(df.columns[11], axis = 1)
df = df.drop(df.columns[10], axis = 1)
df = df.drop(df.columns[6], axis = 1)
df = df.drop(df.columns[2], axis = 1)

#---------------------------------------------------------------
# And repeeat the process
scaler = preprocessing.MinMaxScaler()
datanorm = scaler.fit_transform(df)
dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(transpose(datanorm))
clusters = cluster.hierarchy.linkage(matsim, method = 'complete')
cluster.hierarchy.dendrogram(clusters, color_threshold=20)
plt.show()


df = df.drop(df.columns[7], axis = 1)
df = df.drop(df.columns[6], axis = 1)


df.to_csv("../data/processed/T2_Labels_Gyroscope.csv", index=False)



# ----------------------------------------------------------------
# Apply the same process to the samples

df = pd.read_csv("../data/processed/T2_sample_per1.csv")

# Get only objective data
df = df[[col for col in df.columns if ("Gyroscope" in col and "FFT" not in col) or "labels" == col]]

df = df.drop(df.columns[12], axis = 1)
df = df.drop(df.columns[11], axis = 1)
df = df.drop(df.columns[10], axis = 1)
df = df.drop(df.columns[6], axis = 1)
df = df.drop(df.columns[2], axis = 1)

df = df.drop(df.columns[7], axis = 1)
df = df.drop(df.columns[6], axis = 1)

df.to_csv("../data/processed/T2_sample_per1_gyroscope.csv", index=False)


#------------------------------------------------------------------


df = pd.read_csv("../data/processed/T2_sample_per1_inliers.csv")

df = df[[col for col in df.columns if ("Gyroscope" in col and "FFT" not in col) or "labels" == col]]

df = df.drop(df.columns[12], axis = 1)
df = df.drop(df.columns[11], axis = 1)
df = df.drop(df.columns[10], axis = 1)
df = df.drop(df.columns[6], axis = 1)
df = df.drop(df.columns[2], axis = 1)

df = df.drop(df.columns[7], axis = 1)
df = df.drop(df.columns[6], axis = 1)

df.to_csv("../data/processed/T2_sample_per1_inliers_gyroscope.csv", index=False)
