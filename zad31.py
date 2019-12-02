# Importing libraries
import pandas as pd
import numpy as np
from decimal import Decimal

# Read csv file into a pandas dataframe
df = pd.read_csv("iris_with_errors.csv")

# Take a look at the first few rows
# print (df['sepal.length'].isnull())

print("\nTotal missing values for each column")
print (df.isnull().sum())

print ("\nTotal number of missing values ", df.isnull().sum().sum())

print("\nDetekcja not a number")
#Detecting not numbers
cnt=0
for col in df.columns:
    for row in df[col]:
        if col != "variety":
            try:
                float(row)
            except ValueError:
                print(row, cnt, col)
                df.loc[cnt, col] = 0
                pass
        cnt+=1
    cnt=0

print("\nDetekcja liczb spoza przedziału (0;15)")
# Median
cnt=0
for col in df.columns:
    for row in df[col]:
        if col != "variety":
            if not float(row) > 0 and float(row) < 15:
                print(row)
                median = df[col].median()
                df.loc[cnt, col] = median
        cnt+=1
    cnt=0


print("\nDetekcja napisów")
names = ["Setosa", "Versicolor", "Virginica"]
# Strings
cnt=0
for row in df["variety"]:
    if not row in names:
        print(row)
        if(row.capitalize() in names):
            df.loc[cnt, "variety"] = row.capitalize()
        else:
            for item in names:
                if item[:2] == row[:2]:
                    df.loc[cnt, "variety"] = item
    cnt+=1




# ===================== FIX
for col in df.columns:
    df[col].fillna(125, inplace=True)

print ("\nTotal number of missing values ", df.isnull().sum().sum())