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
from sklearn.metrics import mean_squared_error

# My module import #
sys.path.append("C:/Users/ldl34/PycharmProjects/pythonProject/DNN_STUDY/Module/")
from Module.config import *

# ---------------------------------
# Calculate correlation coefficient(R).
#
def R(OBS, PRE):
    gradient, intercept, r_value, p_value, std_err = stats.linregress(OBS, PRE)
    R_val = r_value

    return R_val

# ---------------------------------
# Calculate Index of Agreement(IOA).
#
def IOA(OBS, AVG_OBS, PRE):
    IOA_up, IOA_down = sum((PRE - OBS)**2), sum((abs(PRE - AVG_OBS) + abs(OBS - AVG_OBS))**2)
    IOA_val = 1 - (IOA_up / IOA_down)

    return IOA_val

# ---------------------------------
# Calculate Root Mean Square Error(RMSE)
#
def RMSE(OBS, PRE):
    MSE = mean_squared_error(OBS, PRE)
    RMSE_val = math.sqrt(MSE)

    return RMSE_val

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


# ---------------------------------
# Network file save.
#
def file_save(epoch, save_weight_path, test_weight_path):
    file1 = save_weight_path+'-'+str(epoch)+'.data-00000-of-00001'
    file2 = save_weight_path+'-'+str(epoch)+'.index'
    file3 = save_weight_path+'-'+str(epoch)+'.meta'

    if os.path.exists(test_weight_path):
        for path_, dirs, files in os.walk(test_weight_path):
            for file in files:
                os.remove(os.path.join(path_, file))

    new_file1 = test_weight_path+'save0.data-00000-of-00001'
    new_file2 = test_weight_path+'save0.index'
    new_file3 = test_weight_path+'save0.meta'

    shutil.move(file1, new_file1)
    shutil.move(file2, new_file2)
    shutil.move(file3, new_file3)

    if os.path.exists(save_weight_path):
        for path_, dirs, files in os.walk(save_weight_path):
            for file in files:
                os.remove(os.path.join(path_, file))


# ---------------------------------
# Network file remove.
#
def file_delete(path):
    if os.path.exists(path):
        for path_, dirs, files in os.walk(path):
            for file in files:
                os.remove(os.path.join(path_, file))


# ---------------------------------
# Data shuffle function.
#
def DataShuffle(Input, Label, NumPattern, Seed_Num):
    np.random.seed(Seed_Num)

    for S in range(NumPattern):

        randomNumber = np.random.randint(0, high=NumPattern, size=2)
        tmp_x = copy.deepcopy(Input[randomNumber[0]])
        tmp_y = copy.deepcopy(Label[randomNumber[0]])

        Input[randomNumber[0]] = copy.deepcopy(Input[randomNumber[1]])
        Label[randomNumber[0]] = copy.deepcopy(Label[randomNumber[1]])

        Input[randomNumber[1]] = copy.deepcopy(tmp_x)
        Label[randomNumber[1]] = copy.deepcopy(tmp_y)

    return Input, Label

# ---------------------------------
# Batch function.
#
def Nextbatch(xdata, ydata, xcol_num, ycol_num, size, cru, length, random=False):
    tmp_x, tmp_y = np.zeros((size, xcol_num)), np.zeros((size, ycol_num))
    index = 0

    if size > (length - cru):
        size = cru

    if size == length:
        print("")

    if random:
        randomIndex = np.random.randint(0, high=length, size=size)
        for S in range(size) :
            tmp_x[i] = copy.deepcopy(xdata[randomIndex[S]])
            tmp_y[i] = copy.deepcopy(ydata[randomIndex[S]])

    elif not random:
        for S in range(cru, (cru + size), 1):
            tmp_x[index] = copy.deepcopy(xdata[S])
            tmp_y[index] = copy.deepcopy(ydata[S])
            index += 1

    return tmp_x, tmp_y


# ---------------------------------
# Early stop condition.
# 'set._step' & 'self._loss' is very important
#
class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')    # bigger: 0, smaller: float('inf') 
        self.patience  = patience
        self.verbose = verbose

    def validate(self, loss, FILE):
        if self._loss < loss :    # bigger: > , smaller: < 
            self._step += 1
            print("Early stopping function step : {}".format(self._step), file=FILE)

            if self._step >= self.patience:
                print("Early stopping function step : {}".format(self._step), file=FILE)

                if self.verbose:
                    return True

        else:
            print("self._loss < loss False : _loss={} // loss={} // Step={}".format(self._loss, loss, self._step), file=FILE)
            self._step = 0
            self._loss = loss

        return False