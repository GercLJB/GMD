# ----------------------------------------
# Import library
#
import os
import sys

# My module import #
sys.path.append(os.path.join(os.path.abspath(os.path.dirname('__main__'))))
from Module.config import *
from Module.PostModule import *

# ----------------------------------------
# Path setting.
#
ABS_PATH = os.path.abspath(os.path.dirname('__main__'))
TIME_PATH = os.path.join(ABS_PATH, "T_RESULT/TIME/")
DAY_PATH = os.path.join(ABS_PATH, "T_RESULT/DAY/")

print("===============>> Start 'T_INDEX_ASS'")
T_INDEX_ASS(IN_PATH=TIME_PATH, OUT_PATH=TIME_PATH, RUN_TIME=RUN_TIME, MATTER=PM, AREA=AREA, TIME_LIST=TIME_LIST[RUN_TIME])
print("===============>> Finish 'T_INDEX_ASS'")

print("===============>> Start 'T_STT_ASS'")
T_STT_ASS(IN_PATH=TIME_PATH, OUT_PATH=TIME_PATH, RUN_TIME=RUN_TIME, MATTER=PM, AREA=AREA, TIME_LIST=TIME_LIST[RUN_TIME])
print("===============>> Finish 'T_STT_ASS'")

print("===============>> Start 'DAY_AVERAGE'")
DAY_AVERAGE(IN_PATH=TIME_PATH, OUT_PATH=DAY_PATH, RUN_TIME=RUN_TIME, MATTER=PM, AREA=AREA)
print("===============>> Finish 'DAY_AVERAGE'")

print("===============>> Start 'D_INDEX_ASS'")
D_INDEX_ASS(IN_PATH=DAY_PATH, OUT_PATH=DAY_PATH, RUN_TIME=RUN_TIME, MATTER=PM, AREA=AREA)
print("===============>> Finish 'D_INDEX_ASS'")

print("===============>> Start 'D_STT_ASS'")
D_STT_ASS(IN_PATH=DAY_PATH, OUT_PATH=DAY_PATH, RUN_TIME=RUN_TIME, MATTER=PM, AREA=AREA)
print("===============>> Finish 'D_STT_ASS'")
print("\n")
