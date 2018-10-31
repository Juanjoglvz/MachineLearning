import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing 
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn import metrics

# Read the data and load it into memory
df_T = pd.read_csv("../../data/processed/T2_Accelerometer.csv")


# Principal Component Analysis
#1 Scalation

scaler = preprocessing.MinMaxScaler()
datanorm = scaler.fit_transform(df_T)

#2 Modelling (PCA - 2 components)

n_components = 2
estimator = PCA (n_components)
X_pca = estimator.fit_transform(datanorm)

# is it representative?
print (estimator.explained_variance_ratio_)



# Plot the PCA result
x = X_pca[:,0]
y = X_pca[:,1]
plt.scatter(x,y)
plt.savefig("../../reports/figures/PCA_Plot_Accelerometer_Day2_Selected")
plt.show()



# Clustering: KMEANS
# First, we try to determine the best number of centroids
# parameters
init = 'k-means++' # initialization method 

# to run 10 times with different random centroids 
# to choose the final model as the one with the lowest SSE
iterations = 10

# maximum number of iterations for each single run
max_iter = 300 

# controls the tolerance with regard to the changes in the 
# within-cluster sum-squared-error to declare convergence

tol = 1e-08

 # random seed
random_state = 3

distortions = []
silhouettes = []

for i in range(2, 20):
    km = KMeans(i, init, n_init = iterations ,max_iter= max_iter, tol = tol,random_state = random_state)
    labels = km.fit_predict(X_pca)
    distortions.append(km.inertia_)
    silhouettes.append(metrics.silhouette_score(X_pca, labels))
    
    
# Print the Distorsion
plt.plot(range(2,20), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.xticks(np.arange(0, 20, 1.0))
plt.savefig("../../reports/figures/KMeans_Distortion_Accelerometer_Day2_Selected")
plt.show()
# Print the Silouhette
plt.plot(range(2,20), silhouettes , marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette')
plt.xticks(np.arange(0, 20, 1.0))
plt.savefig("../../reports/figures/KMeans_Silhouette_Accelerometer_Day2_Selected")
plt.show()


# The best number of clusters is 4
km = KMeans(4, init, n_init = iterations ,max_iter= max_iter, tol = tol,random_state = random_state)
labels = km.fit_predict(X_pca)
df_T["cluster"] = labels




# Plot the PCA result
x = X_pca[:,0]
y = X_pca[:,1]
plt.scatter(x,y, c = km.labels_)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c = "red")
plt.savefig("../../reports/figures/KMeans_Plot_Accelerometer_Day2_Selected")
plt.show()


# fit the model for outlier detection (default)
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the estimator has no predict,
# decision_function and score_samples methods).
y_pred = clf.fit_predict(df_T)
#n_errors = (y_pred != ground_truth).sum()
X_scores = clf.negative_outlier_factor_

n_outliers = 5
lowest_indices = np.argsort(-X_scores, axis=0)[-1:-1-n_outliers:-1]
inliers = np.delete(X_pca, lowest_indices, axis=0)
outliers = X_pca[lowest_indices]

plt.title("Local Outlier Factor (LOF)")
plt.scatter(inliers[:,0], inliers[:,1])
plt.scatter(outliers[:,0], outliers[:,1], s=65, marker="x")
plt.show()

db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
y_db = db.fit_predict(X_pca)

print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X_pca, y_db))

plt.title("DBSCAN clustering")
plt.scatter(X_pca[y_db==0,0], X_pca[y_db==0,1], c='lightblue', marker='o', s=40, label='cluster 1')
plt.scatter(X_pca[y_db==1,0], X_pca[y_db==1,1], c='red', marker='s', s=40, label='cluster 2')
plt.legend()
plt.show()
