import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
plt.savefig("../../reports/figures/PCA_Plot_Accelerometer_Day2_Selected")
plt.show()