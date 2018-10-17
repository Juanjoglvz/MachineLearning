import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA

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
plt.show()


# Clustering: KMEANS
# First, we try to determine the best number of centroids
# parameters
init = 'random' # initialization method 

# to run 10 times with different random centroids 
# to choose the final model as the one with the lowest SSE
iterations = 10

# maximum number of iterations for each single run
max_iter = 300 

# controls the tolerance with regard to the changes in the 
# within-cluster sum-squared-error to declare convergence

tol = 1e-08

 # random seed
random_state = 0

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
plt.show()
# Print the Silouhette
plt.plot(range(2,20), silhouettes , marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silohouette')
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
plt.show()
