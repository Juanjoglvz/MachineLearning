# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from numpy import corrcoef, transpose, arange
from pylab import pcolor, show, colorbar, xticks, yticks, savefig
import sklearn.neighbors
from scipy import cluster

OUTPUT_FIG = "../../reports/figures/2710/"

# Read the data and load it into memory
df = pd.read_csv("../../data/raw/T2.csv")

# Get only the first day of data
df['TimeStemp'] = pd.to_datetime(df['TimeStemp'])
# extract date from datetime
df['date'] = [d.date() for d in df['TimeStemp']]
# list the available days
df['date'].unique()
#filter data by date
df = df[(df['TimeStemp'] > '2016-04-28 00:00:00') & (df['TimeStemp'] <= '2016-04-28 23:59:59')]


# Remove non-numerical columns
df = df.drop(['UserID', 'UUID', 'Version', 'TimeStemp'], axis=1)

for column in df.columns:  # Remove columns with all null values
    if df[column].isnull().all():
        df = df.drop(column, axis=1)
        
df = df.dropna()  # Remove rows with null values

# We remove All the middle samples and all the columns related to Fast Fourier Transformation
df_Nofft = df[[c for c in df if "FFT" not in c and "MIDDLE_SAMPLE" not in c]]

# We take only the features related to the Gyroscope and the Rotation Vector

df_T = df_Nofft[[c for c in df_Nofft if "GyroscopeStat" in c or "RotationVector" in c]]

# We are now going to try and reduce the dimensionality a bit more: Let's see the correlation between variables
R = corrcoef(transpose(df_T))
pcolor(R)
colorbar()
yticks(arange(0,18), range(0,18))
xticks(arange(0,18), range(0,18))
savefig(OUTPUT_FIG + "Features_CorrelationMatrix_preDrop")
show()    

#Similarly to earlier experiments, we can safely remove most of the median values,
#  as there is high correlation between Mean and Median. 

# X axis of the giroscope is the exception. The correlation is not high enough, 
#  so we cannot remove either mean or median safely

# Also, Y mean and X Mean have a surprisingly high correlation. So, we remove y
# axis mean values.

df_T = df_T.drop("GyroscopeStat_y_MEDIAN", axis = 1)
df_T = df_T.drop("GyroscopeStat_z_MEDIAN", axis = 1)
df_T = df_T.drop("RotationVector_xSinThetaOver2_MEDIAN", axis = 1)
df_T = df_T.drop("RotationVector_ySinThetaOver2_MEAN", axis = 1)
df_T = df_T.drop("RotationVector_ySinThetaOver2_MEDIAN", axis = 1)
df_T = df_T.drop("RotationVector_zSinThetaOver2_MEDIAN", axis = 1)

# Let's check the results again

R = corrcoef(transpose(df_T))
pcolor(R)
colorbar()
yticks(arange(0,12),range(0,12))
xticks(arange(0,12),range(0,12))
savefig(OUTPUT_FIG + "Features_CorrelationMatrix_postDrop")
show()    

# Now we plot a dendogram.
dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(transpose(df_T))
clusters = cluster.hierarchy.linkage(matsim, method = 'complete')
cluster.hierarchy.dendrogram(clusters, color_threshold=0)
plt.savefig(OUTPUT_FIG +"Features_Dendogram")
plt.show()

#As expected, the most similar features, the first action taken by the dendogram,
# is grouping features 0 and 1, Gyroscope X Mean and Median. But, we don't want to
# remove any of them, they had a (surprisingly) low correlation.

# Save processed data 
df_T.to_csv('../../data/processed/T2_Gyroscope.csv', index=False)
