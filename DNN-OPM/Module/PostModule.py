# ----------------------------------------
# import library
#
import os
import sys
import math
import shutil
import numpy as np
import pandas as pd
import datetime as d
from scipy import stats
from sklearn.metrics import mean_squared_error

# My module import #
from Module.utile import *
from Module.config import *

# ----------------------------------------
# Calculate ACC, HIT, POD, FAR, F1_SCORE - Day & time result
#
def INDEX_MATRIX(Index_df, Matrix, Bool, write_file):
    Target = Index_df['Target'].tolist()
    if Bool == 'AI':
        Model = Index_df[Bool].tolist()
    else:
        Model = Index_df[Bool].tolist()

    for i in range(len(Index_df)):
        for j in range(4):
            for k in range(4):
                if Target[i] == j + 1 and Model[i] == k + 1:
                    Matrix[j][k] += 1

    for j in range(5):
        Matrix[4][j] = Matrix[0][j] + Matrix[1][j] + Matrix[2][j] + Matrix[3][j]
        Matrix[j][4] = Matrix[j][0] + Matrix[j][1] + Matrix[j][2] + Matrix[j][3]

    # ACC : Accuracy #
    try:
        ACC = ((Matrix[0][0] + Matrix[1][1] + Matrix[2][2] + Matrix[3][3]) / Matrix[4][4]) * 100
        write_file.write("{:.1f}".format(ACC) + "\t" +
                         "{}//{}".format((Matrix[0][0] + Matrix[1][1] + Matrix[2][2] + Matrix[3][3]), Matrix[4][4]) + "\t")
    except ZeroDivisionError:
        ACC = 0
        write_file.write("0.0" + "\t" + "0.0" + "\t")
        pass

    # POD : Probability of Detection # 
    try:
        POD = ((Matrix[2][2] + Matrix[2][3] + Matrix[3][2] + Matrix[3][3]) / (Matrix[2][4] + Matrix[3][4])) * 100
        write_file.write("{:.1f}".format(POD) + "\t" +
                         "{}//{}".format((Matrix[2][2] + Matrix[2][3] + Matrix[3][2] + Matrix[3][3]),(Matrix[2][4] + Matrix[3][4])) + "\t")
    except ZeroDivisionError:
        POD = 0
        write_file.write("0.0" + "\t" + "0.0" + "\t")
        pass

    # FAR : False Alarm Rate #
    try:
        FAR = ((Matrix[0][2] + Matrix[0][3] + Matrix[1][2] + Matrix[1][3]) /
               (Matrix[0][2] + Matrix[0][3] + Matrix[1][2] + Matrix[1][3] + Matrix[2][2] +
                Matrix[2][3] + Matrix[3][2] + Matrix[3][3])) * 100
        write_file.write("{:.1f}".format(FAR) + "\t" +
                         "{}//{}".format((Matrix[0][2] + Matrix[0][3] + Matrix[1][2] + Matrix[1][3]),
                                         (Matrix[0][2] + Matrix[0][3] + Matrix[1][2] + Matrix[1][3] +
                                          Matrix[2][2] + Matrix[2][3] + Matrix[3][2] + Matrix[3][3])) + "\t")
    except ZeroDivisionError:
        FAR = 0
        write_file.write("0.0" + "\t" + "0//0" + "\t")
        pass

    # F1_SCORE : Harmony mean #
    try:
        F1_SCORE = ((2 * (POD * (100 - FAR))) / (POD + (100 - FAR)))

        if (Bool=='AI'): write_file.write("{:.2f}".format(F1_SCORE) + "\n")
        else: write_file.write("{:.2f}".format(F1_SCORE) + "\n")
    except ZeroDivisionError:
        if (Bool == 'AI'): write_file.write("0.0" + "\n")
        else: write_file.write("0.0" + "\n")
        pass

# ----------------------------------------
# Index agreement calculation
# Best network result
def T_INDEX_ASS(IN_PATH, OUT_PATH, RUN_TIME, MATTER, AREA, TIME_LIST):
    I_PATH = IN_PATH
    O_PATH = OUT_PATH

    f = open(O_PATH + "TIME_INDEX_RESULT.txt", "w")
    f.write("Site" + "\t" + "Time&Day" + "\t" +
            "AI_ACC" + "\t" + "Ratio" + "\t" +
            "AI_POD" + "\t" + "Ratio" + "\t" + "AI_FAR" + "\t" + "Ratio" + "\t" + "AI_F1" + "\n")

    for T in TIME_LIST:
        df = pd.read_csv(I_PATH + "T{:02d}_Network.txt".format(T), delimiter="\t")
        f.write("{}".format(AREA) + "\t" + "T{:02d}".format(T) + "\t")

        for col in df.columns:
            if col in ['Target', 'AI']:
                df[col] = df[col].mask(df[col] < RANGE[MATTER]['Good'], other=1)
                df[col] = df[col].mask((df[col] >= RANGE[MATTER]['Good']) & (df[col] < RANGE[MATTER]['Moderate']), other=2)
                df[col] = df[col].mask((df[col] >= RANGE[MATTER]['Moderate']) & (df[col] < RANGE[MATTER]['Bad']), other=3)
                df[col] = df[col].mask(df[col] >= RANGE[MATTER]['Bad'], other=4)

        AI_matrix = [[0 for i in range(5)] for j in range(5)]
        INDEX_MATRIX(Index_df=df, Matrix=AI_matrix, Bool='AI', write_file=f)

        print(":: Index agreement calculation finish, T{:02d} ::".format(T))

    f.close()


def DAY_AVERAGE(IN_PATH, OUT_PATH, RUN_TIME, MATTER, AREA):

    I_PATH = IN_PATH
    O_PATH = OUT_PATH

    if not os.path.isdir(O_PATH): os.makedirs(O_PATH)

    IN_File = os.listdir(I_PATH)
    IN_File = [file for file in IN_File if file.endswith("_Network.txt")]

    df_T05 = pd.read_csv(I_PATH + "/T05_Network.txt", delimiter="\t")
    df_T06 = pd.read_csv(I_PATH + "/T06_Network.txt", delimiter="\t")
    df_T07 = pd.read_csv(I_PATH + "/T07_Network.txt", delimiter="\t")
    df_T08 = pd.read_csv(I_PATH + "/T08_Network.txt", delimiter="\t")
    df_T09 = pd.read_csv(I_PATH + "/T09_Network.txt", delimiter="\t")
    df_T10 = pd.read_csv(I_PATH + "/T10_Network.txt", delimiter="\t")
    df_T11 = pd.read_csv(I_PATH + "/T11_Network.txt", delimiter="\t")
    df_T12 = pd.read_csv(I_PATH + "/T12_Network.txt", delimiter="\t")
    df_T13 = pd.read_csv(I_PATH + "/T13_Network.txt", delimiter="\t")
    df_T14 = pd.read_csv(I_PATH + "/T14_Network.txt", delimiter="\t")
    df_T15 = pd.read_csv(I_PATH + "/T15_Network.txt", delimiter="\t")

    D0 = pd.DataFrame([], columns=df_T07.columns)
    D1 = pd.DataFrame([], columns=df_T08.columns)
    D2 = pd.DataFrame([], columns=df_T12.columns)
    D0["Date"] = df_T07["Date"]
    D1["Date"] = df_T08["Date"]
    D2["Date"] = df_T12["Date"]

    for col in ["Target", "AI"]:
        D0[col] = (df_T05[col] + df_T06[col] + df_T07[col]) / 3
        D1[col] = (df_T08[col] + df_T09[col] + df_T10[col] + df_T11[col]) / 4
        D2[col] = (df_T12[col] + df_T13[col] + df_T14[col] + df_T15[col]) / 4

    D0 = D0.round(decimals=0)
    D1 = D1.round(decimals=0)
    D2 = D2.round(decimals=0)
    D0.to_csv(O_PATH + "D+0_avg.txt", sep="\t", index=False)
    D1.to_csv(O_PATH + "D+1_avg.txt", sep="\t", index=False)
    D2.to_csv(O_PATH + "D+2_avg.txt", sep="\t", index=False)


def D_INDEX_ASS(IN_PATH, OUT_PATH, RUN_TIME, MATTER, AREA):

    I_PATH = IN_PATH
    O_PATH = OUT_PATH
    Good, Moderate, Bad = 15.5, 35.5, 75.5

    f = open(I_PATH + "DAY_INDEX_RESULT.txt", "w")
    f.write("Site" + "\t" + "Time&Day" + "\t" +
            "AI_ACC" + "\t" + "Ratio" + "\t" + "AI_HIT" + "\t" + "Ratio" + "\t" +
            "AI_POD" + "\t" + "Ratio" + "\t" + "AI_FAR" + "\t" + "Ratio" + "\t" + "AI_F1" + "\n")

    for D in range(0, 3):
        df = pd.read_csv(I_PATH + "D+{}_avg.txt".format(D), delimiter="\t")
        f.write("{}".format(AREA) + "\t" + "D+{}".format(D) + "\t")

        for col in df.columns:
            if col in ['Target', 'AI']:
                df[col] = df[col].mask(df[col] < RANGE[MATTER]['Good'], other=1)
                df[col] = df[col].mask((df[col] >= RANGE[MATTER]['Good']) & (df[col] < RANGE[MATTER]['Moderate']), other=2)
                df[col] = df[col].mask((df[col] >= RANGE[MATTER]['Moderate']) & (df[col] < RANGE[MATTER]['Bad']), other=3)
                df[col] = df[col].mask(df[col] >= RANGE[MATTER]['Bad'], other=4)

        AI_matrix = [[0 for i in range(5)] for j in range(5)]

        INDEX_MATRIX(Index_df=df, Matrix=AI_matrix, Bool='AI', write_file=f)
        print(":: Day index agreement calculation finish, D+{} ::".format(D))

    f.close()


# ----------------------------------------
# Time statistic calculation.
#
def T_STT_ASS(IN_PATH, OUT_PATH, RUN_TIME, MATTER, AREA, TIME_LIST):

    I_PATH = IN_PATH
    O_PATH = OUT_PATH

    f = open(O_PATH + "TIME_STT_RESULT.txt", "w")
    f.write("Site" + "\t" + "Time&Day" + "\t" +
            "AVG_OBS" + "\t" + "AVG_AI" + "\t" + "AI_R" + "\t" + "AI_IOA" + "\t" + "AI_RMSE" + "\n")

    for T in TIME_LIST:
        df = pd.read_csv(I_PATH + "T{:02d}_Network.txt".format(T), delimiter="\t")

        OBS = df['Target']
        AI = df['AI']

        AVG_OBS, AVG_AI = (sum(OBS) / len(df)), (sum(AI) / len(df))

        f.write("{}".format(AREA) + "\t" + "T{:02d}".format(T) + "\t" +
                "{:.1f}".format(AVG_OBS) + "\t" + "{:.1f}".format(AVG_AI) + "\t" +
                "{:.2f}".format(R(OBS, AI)) + "\t" + "{:.2f}".format(IOA(OBS, AVG_OBS, AI)) + "\t" + "{:.1f}".format(RMSE(OBS, AI)) + "\n")

        print(":: Time statistic calculation finish, T{:02d} ::".format(T))

    f.close()


# ----------------------------------------
# Day statistic calculation
#
def D_STT_ASS(IN_PATH, OUT_PATH, RUN_TIME, MATTER, AREA):

    I_PATH = IN_PATH
    O_PATH = OUT_PATH

    f = open(O_PATH + "DAY_STT_RESULT.txt", "w")
    f.write("Site" + "\t" + "Time&Day" + "\t" +
            "AVG_OBS" + "\t" + "AVG_AI" + "\t" + "AI_R" + "\t" + "AI_IOA" + "\t" + "AI_RMSE" + "\n")

    for D in range(0, 3):
        df = pd.read_csv(I_PATH + "D+{}_avg.txt".format(D), delimiter="\t")

        OBS = df['Target']
        AI = df['AI']

        AVG_OBS, AVG_AI = (sum(OBS) / len(df)), (sum(AI) / len(df))

        f.write("{}".format(AREA) + "\t" + "D+{}".format(D) + "\t" +
                "{:.1f}".format(AVG_OBS) + "\t" + "{:.1f}".format(AVG_AI) + "\t" +
                "{:.2f}".format(R(OBS, AI)) + "\t" + "{:.2f}".format(IOA(OBS, AVG_OBS, AI)) + "\t" + "{:.1f}".format(RMSE(OBS, AI)) + "\n")

        print(":: Day statistic calculation finish, D+{} ::".format(D))

    f.close()