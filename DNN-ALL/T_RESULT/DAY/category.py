import os
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


for D in range(0,3):
    df = pd.read_csv("./D+{}_avg.txt".format(D), delimiter="\t")

    for col in df.columns:
        if col in ['Target', 'AI', 'CMAQ']:
            df[col] = df[col].mask(df[col] < 15.5, other=1)
            df[col] = df[col].mask((df[col] >= 15.5) & (df[col] < 35.5), other=2)
            df[col] = df[col].mask((df[col] >= 35.5) & (df[col] < 75.5), other=3)
            df[col] = df[col].mask(df[col] >= 75.5, other=4)

    y = df['Target']
    p = df['AI']
    p_cmaq = df['CMAQ']

    print("AI :: {}".format(D))
    print(classification_report(y, p, target_names=['Good', 'Moderate', 'Bad', 'VeryBad']))
    print("CMAQ :: {}".format(D))
    print(classification_report(y, p_cmaq, target_names=['Good', 'Moderate', 'Bad', 'VeryBad']))
    print("\n")