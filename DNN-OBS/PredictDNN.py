# ----------------------------------------
# import library.
#
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf

# My module import #
sys.path.append(os.path.join(os.path.abspath(os.path.dirname('__main__'))))
from Module.utile import *
from Module.config import *

# ----------------------------------------
# Hyper-parameter setting
#
INDIM, HIDDIM1, HIDDIM2, HIDDIM3, HIDDIM4, OUTDIM = 26, 13, 7, 3, 2, 1

# ----------------------------------------
# ABS path setting.
#
ABS_PATH = os.path.abspath(os.path.dirname('__main__'))
INPUT_PATH = os.path.join(ABS_PATH, "INPUT/")
TEST_NETWORK_PATH = os.path.join(ABS_PATH, "TEST_NETWORK/net_save/{{T_TIME}}/")
RESULT_PATH = os.path.join(ABS_PATH, "T_RESULT/TIME/")

# ----------------------------------------
# Building graph
# Name scope is 'NIER_DNN'
#
with tf.name_scope('NIER_DNN'):
    tf.set_random_seed(1)
    X = tf.placeholder(tf.float32, shape=[None, INDIM], name="GraphInput")
    Y = tf.placeholder(tf.float32, shape=[None, OUTDIM], name="GraphTarget")
    # Layer1 #
    W1 = tf.Variable(tf.random_uniform([INDIM, HIDDIM1], minval=-1.0, maxval=1.0) / tf.sqrt(float(INDIM)), name="L1_W")
    B1 = tf.Variable(tf.random_uniform([HIDDIM1], minval=-0.5, maxval=0.5), name="L1_B")
    Layer1 = tf.sigmoid(tf.matmul(X, W1) + B1, name="Layer1")
    # Layer2 #
    W2 = tf.Variable(tf.random_uniform([HIDDIM1, HIDDIM2], minval=-1.0, maxval=1.0) / tf.sqrt(float(HIDDIM1)), name="L2_W")
    B2 = tf.Variable(tf.random_uniform([HIDDIM2], minval=-0.5, maxval=0.5), name="L2_B")
    Layer2 = tf.sigmoid(tf.matmul(Layer1, W2) + B2, name="Layer2")
    # Layer3 #
    W3 = tf.Variable(tf.random_uniform([HIDDIM2, HIDDIM3], minval=-1.0, maxval=1.0) / tf.sqrt(float(HIDDIM2)), name="L3_W")
    B3 = tf.Variable(tf.random_uniform([HIDDIM3], minval=-0.5, maxval=0.5), name="L3_B")
    Layer3 = tf.sigmoid(tf.matmul(Layer2, W3) + B3, name="Layer3")
    # Layer4 #
    W4 = tf.Variable(tf.random_uniform([HIDDIM3, HIDDIM4], minval=-1.0, maxval=1.0) / tf.sqrt(float(HIDDIM3)), name="L4_W")
    B4 = tf.Variable(tf.random_uniform([HIDDIM4], minval=-0.5, maxval=0.5), name="L4_B")
    Layer4 = tf.sigmoid(tf.matmul(Layer3, W4) + B4, name="Layer4")
    # Layer5 #
    W5 = tf.Variable(tf.random_uniform([HIDDIM4, OUTDIM], minval=-1.0, maxval=1.0) / tf.sqrt(float(HIDDIM4)), name="L5_W")
    B5 = tf.Variable(tf.random_uniform([OUTDIM], minval=-0.5, maxval=0.5), name="L5_B")
    OutLayer = tf.matmul(Layer4, W5) + B5

    hypothesis = tf.sigmoid(OutLayer, name="Hypothesis")

# ----------------------------------------
# Predict.
#
sess = tf.Session()
saver = tf.train.Saver()
for T in TIME_LIST[RUN_TIME]:
    W_PATH = TEST_NETWORK_PATH.replace("{{T_TIME}}", 'T{:02d}'.format(T))
    R_PATH = RESULT_PATH.replace("{{T_TIME}}", 'T{:02d}'.format(T))
    Make_folder(R_PATH)

    # Laod data #
    Validation_df = pd.read_csv(INPUT_PATH + "T{:02d}_TEST.txt".format(T), delimiter="\t")
    Validation_x = Validation_df.loc[:, Validation_df.columns != 'Target'].drop(labels=['Date'], axis=1)
    Validation_x_lis, Validation_y = Validation_x.values.tolist(), Validation_df['Target'].tolist()

    # Load scale information data #
    SCALE_df = pd.read_csv(INPUT_PATH + "T{:02d}_SCALE_INFO.txt".format(T), delimiter="\t").set_index('Variable')
    PM25_MIN, PM25_MAX, PM25_MEAN, PM25_STD = SCALE_df.loc['Target'][2], SCALE_df.loc['Target'][3], SCALE_df.loc['Target'][1], SCALE_df.loc['Target'][0]

    # Restore netowkr #
    saver.restore(sess, W_PATH + 'save0')
    Result = sess.run(hypothesis, feed_dict={X: Validation_x_lis})

    # Inverse :: Min-Max scaler & Normalization #
    Result_df = pd.DataFrame([], columns=['Date', 'Target', 'AI'])
    Result_df['Date'] = Validation_df['Date']
    Result_df['Target'] = (Validation_df['Target'] * (PM25_MAX - PM25_MIN)) + PM25_MIN
    Result_df['Target'] = (Result_df['Target'] * PM25_STD) + PM25_MEAN

    # Predict PM2.5 #
    AI_list = np.zeros(shape=len(Validation_df), dtype=np.float64).tolist()
    for i in range(len(Validation_df)):
        AI_list[i] = (Result[i][0] * (PM25_MAX - PM25_MIN)) + PM25_MIN
        AI_list[i] = (AI_list[i] * PM25_STD) + PM25_MEAN
    Result_df['AI'] = AI_list

    Result_df = Result_df.round(decimals=0)
    Result_df.to_csv(R_PATH + "T{:02d}_Network.txt".format(T), sep="\t", index=False)

    print("=====> Finish Predict :: T-Step={}".format(T))
print("\n")
