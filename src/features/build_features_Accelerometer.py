# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

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

# We take only the features related to the Accelerometer and the Linear Acceleration

df_T = df_Nofft[[c for c in df_Nofft if "LinearAcceleration" in c or "AccelerometerStat" in c]]

    
# We now remove the covariances

df_T = df_T[[c for c in df_T if "COV" not in c]]

# Save processed data 
df_T.to_csv('../../data/processed/T2_Accelerometer.csv', index=False)
