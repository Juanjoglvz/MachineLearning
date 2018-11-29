
import pandas as pd

uuid = []
with open("../../data/raw/Moriarty.csv") as f, open("../../data/interim/Moriarty.csv", "w") as w:
    f.readline()
    lines = f.readlines()
    for i in lines:
        if "msec" not in i:
            w.write(i.replace(";", " "))
        else:
            i = i.replace(";", " ")
            w.write(i.replace("],", "] "))
        #Avoid last, empty line
        if i != "\n":
            feat = i.split(",")
            uuid.append(int(feat[1]))
            
        
    
df = pd.read_csv("../../data/raw/T2.csv")
df["labels"] = 0



current = 0
for index, row in df.iterrows():
    if (index > 0 and index < len(df)) and current < len(uuid):
        if (df.iloc[index - 1]['UUID'] <= uuid[current]):
            if (row['UUID'] > uuid[current]):
                current = current + 1
                df.loc[index-1, 'labels'] = 1
                while current < len(uuid) and row['UUID'] > uuid[current] :
                    current += 1


   
for column in df.columns:  # Remove columns with all null values
    if df[column].isnull().all():
        df = df.drop(column, axis=1)
        
df = df.dropna()  # Remove rows with null values        



    

