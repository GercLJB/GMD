# ---------------------------------
# Import library.
#
import os
import sys
import numpy as np
import pandas as pd


# ---------------------------------
# Share parameter setting.
#
RUN_TIME, PM, AREA, TARGET = '09', 'PM2.5', 'SU', 'O_PM2_5'
MODEL_NAME = ['DNN-OBS', 'DNN-OPM', 'DNN-ALL']
RANGE = {'PM2.5':{'Good':15.5, 'Moderate':35.5, 'Bad':75.5}}
INDEX_NAME = {'ACC':{'FullName':'Accuracy', 'Unit':'(%)'},
              'POD':{'FullName':'Probability Of Detection', 'Unit':'(%)'},
              'FAR':{'FullName':'False Alarm Rate', 'Unit':'(%)'},
              'F1SCORE':{'FullName':'F1score', 'Unit':'(%)'}}
TIME_LIST = {'09':[i for i in range(5, 16)]}

# Model Training parameter #
MAX_STEP, DisplayStep, batch_size, Stddev = 100000, 100, 32, 0.5
COUNTS = 5

# ---------------------------------
# Share variable list.
#
DATA_TYPE = {'OBS':'obs', 'FORECAST':'forecast'}
OBS_VAL = ['O_U', 'O_V', 'O_Pa', 'O_ta', 'O_td', 'O_RH', 'O_RN_ACC', 'O_radiation', 'O_O3', 'O_NO2', 'O_CO', 'O_SO2', 'O_PM10', 'O_PM2_5']
FORECAST_VAL = ["f_PM10", "f_PM2_5", "f_Ta", "f_Pa", "f_RH", "f_MH", "f_U", "f_V", "f_RN_ACC",
                "f_850hpa_gpm", "f_850hpa_U", "f_850hpa_V", "f_850hpa_RH", "f_850hpa_Ta", "f_925hpa_gpm", "f_925hpa_U", "f_925hpa_V", "f_temp_850_925"]
OBS_DROP = FORECAST_VAL
OPM_DROP = ['f_PM2_5', 'f_PM10']
ALL_DROP = ['f_PM10']