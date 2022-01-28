# ---------------------------------
# Import library
#
import os
import sys
import numpy as np
import pandas as pd

# My module import #
sys.path.append(os.path.abspath(os.path.dirname('__main__')))
from Module.config import *
from Module.utile import *

# ---------------------------------
# Path setting
#
ABS_PATH = os.path.abspath(os.path.dirname('__main__'))
RAW_PATH = os.path.join(ABS_PATH, "RawData/")
RESULT_PATH = os.path.join(ABS_PATH, "{{MODEL}}/INPUT/")


# ---------------------------------
# Data preprocessing.
#
for MN in MODEL_NAME:

    RESULT = RESULT_PATH.replace("{{MODEL}}", MN)
    Make_folder(RESULT)

    # Each variable data file load #
    OBS_df = pd.read_csv(RAW_PATH + "SU_obs_09.txt", delimiter="\t")
    F_df = pd.read_csv(RAW_PATH + "SU_f_09.txt", delimiter="\t")

    # Merge data file #
    MERGE_df = pd.merge(OBS_df, F_df, on=['Date', 'Time'], how="outer")
    if MN == 'DNN-OBS': MERGE_df = MERGE_df.drop(labels=OBS_DROP, axis=1)
    elif MN == 'DNN-OPM': MERGE_df = MERGE_df.drop(labels=OPM_DROP, axis=1)
    elif MN == 'DNN-ALL': MERGE_df = MERGE_df.drop(labels=ALL_DROP, axis=1)

    # Unnecessary data remove{T01 ~ T(OBS_num)} #
    MERGE_df = MERGE_df.loc[MERGE_df.Time >= 4].reset_index(drop=True)

    # Make 'obs T{OBS_num}', 'forecast Tx', 'Target Tx' #
    T_OBS = MERGE_df.loc[MERGE_df['Time'] == 4, OBS_df.columns].reset_index(drop=True)
    T_FORE = MERGE_df.loc[MERGE_df['Time'] > 4].drop(labels=OBS_VAL, axis=1).reset_index(drop=True)
    T_FORE['Target'] = MERGE_df.loc[MERGE_df['Time'] > 4, TARGET].reset_index(drop=True)

    # Drop 'NaN' value of 'obs T{OBS_num}' data #
    for COL in T_OBS.columns:
        if COL in ['Date', 'Time']: pass
        else: T_OBS = T_OBS[(T_OBS[COL] != -9999)].reset_index(drop=True)

    # Drop 'NaN' value of 'forecast Tx' data #
    FINE_NAN = np.unique(np.argwhere((T_FORE.values == -9999))[:, 0])
    NAN_DATE = np.unique(T_FORE['Date'][FINE_NAN])
    DROP_NAN = T_FORE[T_FORE['Date'].isin(NAN_DATE)].index
    T_FORE.drop(DROP_NAN, inplace=True)
    T_FORE = T_FORE.dropna().reset_index(drop=True)

    # Merge data 'obs T{OBS_num}', 'forecast Tx', 'Target Tx' #
    for T in TIME_LIST[RUN_TIME]:
        Tx_OBS = T_OBS.drop(labels='Time', axis=1).reset_index(drop=True)
        Tx_FORE = T_FORE.loc[(T_FORE['Time'] == T)].reset_index(drop=True)
        ALL_df = pd.merge(Tx_OBS, Tx_FORE, on=['Date'], how="inner").reset_index(drop=True)

        # Data split (Learn data & Validation data & Test data) #
        LEARN_df = ALL_df.loc[((ALL_df['Date'] >= 20160101) & (ALL_df['Date'] <= 20181231))].reset_index(drop=True).copy()
        VALI_df = ALL_df.loc[(ALL_df['Date'] >= 20190101) & (ALL_df['Date'] <= 20191231)].reset_index(drop=True).copy()
        TEST_df = ALL_df.loc[(ALL_df['Date'] >= 20210101) & (ALL_df['Date'] <= 20210331)].reset_index(drop=True).copy()

        # Data reformatting - adjust 'Z-score' #
        LEARN_df, SCALE_df = STD_NORMALIZATION(DATA=LEARN_df)
        SCALE_df.to_csv(RESULT + "T{:02d}_SCALE_INFO.txt".format(T), sep="\t", index=False)
        SCALE_df = SCALE_df.set_index('Variable')

        for COL in VALI_df.columns:
            if COL in ['Date', 'Time']: pass

            else:
                MEAN = SCALE_df.loc[COL][1]
                STD = SCALE_df.loc[COL][0]
                VALI_df[COL] = (VALI_df[COL] - MEAN) / STD
                TEST_df[COL] = (TEST_df[COL] - MEAN) / STD

        # Data reformatting - adjust Min-Max Scale #
        L_COLS = np.intersect1d(LEARN_df.columns, SCALE_df.index)
        for COL in L_COLS:
            LEARN_df = SCALER(DATA=LEARN_df, MINMAX=SCALE_df, COLUMNS=COL)
            VALI_df = SCALER(DATA=VALI_df, MINMAX=SCALE_df, COLUMNS=COL)
            TEST_df = SCALER(DATA=TEST_df, MINMAX=SCALE_df, COLUMNS=COL)

        # Append 'Julian Fuzzy' value #
        LEARN_df, VALI_df, TEST_df = FUZZY(LEARN_df), FUZZY(VALI_df), FUZZY(TEST_df)

        # Drop 'Time' column #
        LEARN_df, VALI_df, TEST_df = LEARN_df.drop(labels=['Time'], axis=1), VALI_df.drop(labels=['Time'], axis=1), TEST_df.drop(labels=['Time'], axis=1)

        # Test data reformatting - adjust 'Z-score' & Min-Max Scale #
        LEARN_df.to_csv(RESULT + "T{:02d}_LEARN.txt".format(T), sep="\t", index=False)
        VALI_df.to_csv(RESULT + "T{:02d}_VALI.txt".format(T), sep="\t", index=False)
        TEST_df.to_csv(RESULT + "T{:02d}_TEST.txt".format(T), sep="\t", index=False)

        print("=====> T{:02d} Training & Validation & Test data :: {} & {} & {}".format(T, LEARN_df.shape, VALI_df.shape, TEST_df.shape))
    print("\n")
