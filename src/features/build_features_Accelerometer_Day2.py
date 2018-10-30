# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from numpy import corrcoef, transpose, arange
from pylab import pcolor, show, colorbar, xticks, yticks, savefig
import sklearn.neighbors
from scipy import cluster
from sklearn import preprocessing

# Read the data and load it into memory
df = pd.read_csv("../../data/raw/T2.csv")



# Get only the first day of data
df['TimeStemp'] = pd.to_datetime(df['TimeStemp'])
# extract date from datetime
df['date'] = [d.date() for d in df['TimeStemp']]
# list the available days
df['date'].unique()
#filter data by date
df = df[(df['TimeStemp'] > '2016-05-10 00:00:00') & (df['TimeStemp'] <= '2016-05-10 23:59:59')]


# Remove non-numerical columns
df = df.drop(['UserID', 'UUID', 'Version', 'TimeStemp'], axis=1)

for column in df.columns:  # Remove columns with all null values
    if df[column].isnull().all():
        df = df.drop(column, axis=1)
        
df = df.dropna()  # Remove rows with null values




# We remove All the middle samples and all the columns related to Fast Fourier Transformation
df_Nofft = df[[c for c in df if "FFT" not in c and "MIDDLE_SAMPLE" not in c]]

# We take only the features related to the Accelerometer and the Linear Acceleration

df_T = df_Nofft[[c for c in df_Nofft if "LinearAcceleration" in c or "AccelerometerStat" in c]]

# We are now going to try and reduce the dimensionality a bit more: Let's see the correlation between variables
R = corrcoef(transpose(df_T))
pcolor(R)
colorbar()
yticks(arange(0,24),range(0,24))
xticks(arange(0,24),range(0,24))
savefig("../../reports/figures/Features_CorrelationMatrix_preDrop_Da2")
show()    

# Turns there's always a close-to-1 correlation between the mean values (columns 0, 3, 6) and the
# median values (columns 1, 4, 7), so we erase them:

df_T = df_T.drop("AccelerometerStat_x_MEDIAN", axis = 1)
df_T = df_T.drop("AccelerometerStat_y_MEDIAN", axis = 1)
df_T = df_T.drop("AccelerometerStat_z_MEDIAN", axis = 1)
df_T = df_T.drop("LinearAcceleration_x_MEDIAN", axis = 1)
df_T = df_T.drop("LinearAcceleration_y_MEDIAN", axis = 1)
df_T = df_T.drop("LinearAcceleration_z_MEDIAN", axis = 1)

# This time it seems there isnt as big correlation as the previous day between the means of the different axis
# But we will erase them as well to maintain the same features of the data:

df_T = df_T.drop("AccelerometerStat_x_MEAN", axis = 1)
df_T = df_T.drop("AccelerometerStat_y_MEAN", axis = 1)
df_T = df_T.drop("AccelerometerStat_z_MEAN", axis = 1)

# Let's see the new correlation matrix
R = corrcoef(transpose(df_T))
pcolor(R)
colorbar()
yticks(arange(0,15),range(0,15))
xticks(arange(0,15),range(0,15))
savefig("../../reports/figures/Features_CorrelationMatrix_postDrop_Day2")
show()   

# There is still a high correlation between the first three columns, which are the variances
# of the three axis. Lets see if by doing feature clustering we can see the same relationship.

# Now we plot a dendogram.
scaler = preprocessing.MinMaxScaler()
datanorm = scaler.fit_transform(df_T)

dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(transpose(datanorm))
clusters = cluster.hierarchy.linkage(matsim, method = 'complete')
cluster.hierarchy.dendrogram(clusters, color_threshold=50)
plt.savefig("../../reports/figures/Features_Dendogram_Day2")
plt.show()

# We can see from the dendogram that the values are still very related, so we remove the variances:
df_T = df_T.drop("AccelerometerStat_x_VAR", axis = 1)
df_T = df_T.drop("AccelerometerStat_y_VAR", axis = 1)
df_T = df_T.drop("AccelerometerStat_z_VAR", axis = 1)

# Save processed data 
df_T.to_csv('../../data/processed/T2_Accelerometer.csv', index=False)
