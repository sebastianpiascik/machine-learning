# repl.it/python3
import pandas as pd

df = pd.read_csv("iris.csv")

amountOfTrue = 0

def myPredictRow(sl,sw,pl,pw):
    if pw < 1:
        return 'setosa'
    else:
        if pl >= 5:
            return 'virginica'
        else:
            return 'versicolor'

for index, row in df.iterrows():
    sl = row['sepallength']
    sw = row['sepalwidth']
    pl = row['petallength']
    pw = row['petalwidth']
    flowerClass = myPredictRow(sl, sw, pl, pw)
    # print(index, flowerClass, row['class'])
    if flowerClass in row['class']:
        amountOfTrue = amountOfTrue + 1

print("Ilość zgadniętych: ", amountOfTrue)
print("Procent: ", amountOfTrue/150)
