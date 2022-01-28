# ----------------------------------------
# import library
#
import os
import sys
import copy
import math
import shutil
import numpy as np
import pandas as pd
import datetime as d
from scipy import stats

# My module import #
from Module.config import *


# ---------------------------------
# Make new folder. 
#
def Make_folder(PATH):
    if not os.path.isdir(PATH):
        os.makedirs(PATH)

def Mkdir(PATH):
    if not os.path.isdir(PATH):
        os.makedirs(PATH)


# ---------------------------------
# Reformatting date.
#
def ReDate(DATE_LIST):
    Y_LIST, M_LIST, D_LIST, J_LIST = [], [], [], []

    for DATE in range(len(DATE_LIST)):
        Y_LIST.append(int(str(DATE_LIST[DATE])[0:4]))
        M_LIST.append(int(str(DATE_LIST[DATE])[4:6]))
        D_LIST.append(int(str(DATE_LIST[DATE])[6:8]))
        J_LIST.append(Juldate(YEAR=Y_LIST[DATE], MONTH=M_LIST[DATE], DAY=D_LIST[DATE]))

    return Y_LIST, M_LIST, D_LIST, J_LIST

# ---------------------------------
# Reformatting date.
#
def TimeSeriesDateFormat(DateList):
    FromDateTime = d.datetime.strptime(str(int(DateList[0])), "%Y%m%d")
    ToDateTime = d.datetime.strptime(str(int(DateList[-1])), "%Y%m%d")

    ExistsDateList = []
    NowDateTime = FromDateTime
    while NowDateTime <= ToDateTime:
        YYYY = str(NowDateTime.year).zfill(4)
        MM = str(NowDateTime.month).zfill(2)
        DD = str(NowDateTime.day).zfill(2)
        StrFormat = str("{}-{}-{}".format(YYYY, MM, DD))
        TimeFormat = d.datetime.strptime(StrFormat, "%Y-%m-%d")
        ExistsDateList.append(TimeFormat)
        NowDateTime = NowDateTime + d.timedelta(days=1)

    return ExistsDateList


# ---------------------------------
# Adjust 'Z-score' normalization.
# X_nor = {X_ori - mean(X)} / Std(X)
#
def STD_NORMALIZATION(DATA):
    INFO_df = pd.DataFrame([], columns=['Variable', 'Std', 'Mean', 'ZMin', 'ZMax'])
    COL_NAME, STD, MEAN, MIN, MAX = [], [], [], [], []

    for COL in DATA.columns:

        if COL in ['Date', 'Time']:
            pass
        else:
            DATA_MEAN = sum(DATA[COL]) / len(DATA)
            DATA_STD = DATA[COL].std()

            DATA[COL] = (DATA[COL] - DATA_MEAN) / DATA_STD

            ZMIN, ZMAX = min(DATA[COL]), max(DATA[COL])

            COL_NAME.append(COL)
            STD.append(DATA_STD)
            MEAN.append(DATA_MEAN)
            MIN.append(ZMIN)
            MAX.append(ZMAX)

    INFO_df['Variable'], INFO_df['Std'], INFO_df['Mean'], INFO_df['ZMin'], INFO_df['ZMax'] = COL_NAME, STD, MEAN, MIN, MAX

    return DATA, INFO_df


# ---------------------------------
# Adjust 'Min-Max scaler'.
# X_new = {X_ori - min(X_ori)} / {min(X_ori) - max(X_ori)}
# Range : 0 ~ 1
#
def SCALER(DATA, MINMAX, COLUMNS):
    if COLUMNS in DATA.columns:
        DATA.loc[(DATA[COLUMNS] < MINMAX.loc[COLUMNS][2]), COLUMNS] = MINMAX.loc[COLUMNS][2]
        DATA.loc[(DATA[COLUMNS] > MINMAX.loc[COLUMNS][3]), COLUMNS] = MINMAX.loc[COLUMNS][3]

    DATA[COLUMNS] = (DATA[COLUMNS] - MINMAX.loc[COLUMNS][2]) / (MINMAX.loc[COLUMNS][3] - MINMAX.loc[COLUMNS][2])

    return DATA


# ---------------------------------
# Generation fuzzy value. (M01 ~ M12)
#
def FUZZY(DATA):
    DATE = pd.Series(DATA.Date, dtype=int).values
    DATA_MONTH = np.zeros((len(DATE), 12), dtype=np.float64)

    for ptn, i in enumerate(DATE):
        M, D = int(str(i)[4:6]), int(str(i)[6:])

        if (D < 15) and (M - 1 == 0) :
            M_ADJ = 12
        elif (D > 15) and (M + 1 == 13):
            M_ADJ = 1
        elif (D < 15):
            M_ADJ = M - 1
        elif (D > 15):
            M_ADJ = M + 1

        if (D < 15):
            temp = (D / 28.0) + (13.0 / 28.0)
            DATA_MONTH[ptn][M - 1] = temp
            DATA_MONTH[ptn][M_ADJ - 1] = 1 - temp
        elif (D > 15):
            temp = 1.5 - (D / 30.0)
            DATA_MONTH[ptn][M - 1] = temp
            DATA_MONTH[ptn][M_ADJ - 1] = 1 - temp
        elif (D == 15):
            DATA_MONTH[ptn][M - 1] = 1.0

    COL_NAME = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10', 'M11', 'M12']
    JULIAN_FUZZY = pd.DataFrame(DATA_MONTH, columns=COL_NAME)
    APPEND_DATA = DATA.loc[:, 'O_U':'Target']

    MERGE = pd.concat([DATA.Date, JULIAN_FUZZY, APPEND_DATA], axis=1)

    return MERGE


# ---------------------------------
# Data split is Target value & lean value.
#
def Data_Split(DATA):
    DATA_X = DATA.loc[:, DATA.columns != 'Target'].drop(labels=['Date'], axis=1)
    DATA_X_LIST = DATA_X.values.tolist()
    DATA_Y = DATA['Target'].tolist()
    DATA_LENGTH = len(DATA)

    return DATA_X_LIST, DATA_Y, DATA_LENGTH