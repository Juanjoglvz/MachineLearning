# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv("../../data/raw/T2.csv")
df = df[0:2838]

# Remove non-numerical columns
df = df.drop(['UserID', 'Version', 'TimeStemp'], axis=1)

for column in df.columns:  # Remove columns with all null values
    if df[column].isnull().all():
        print(column)
        df = df.drop(column, axis=1)
        
df = df.dropna()  # Remove rows with null values

df.to_csv('../../data/interim/T2_processed.csv', index=False)
