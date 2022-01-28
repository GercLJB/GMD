# ---------------------------------
# Import library
#
import os
import sys
import time
import copy
import numpy as np
import pandas as pd
import datetime as d
import tensorflow as tf

# My module import #
sys.path.append(os.path.join(os.path.abspath(os.path.dirname('__main__'))))
from Module.utile import *
from Module.config import *

# ---------------------------------
# ABS path setting.
#
ABS_PATH = os.path.abspath(os.path.dirname('__main__'))
INPUT_PATH = os.path.join(ABS_PATH, "INPUT/")
TEST_NETWORK_PATH = os.path.join(ABS_PATH, "TEST_NETWORK/net_save/{{T_TIME}}/")
SAVE_NETWORK_PATH = os.path.join(ABS_PATH, "SAVE_NETWORK/net_save/{{T_TIME}}/")
LOG_PATH = os.path.join(ABS_PATH, "LOG/")
Mkdir(LOG_PATH)

# ---------------------------------
# Building.
#
INDIM, HIDDIM1, HIDDIM2, HIDDIM3, HIDDIM4, OUTDIM = 42, 20, 10, 5, 3, 1
with tf.name_scope('NIER_DNN'):
    tf.set_random_seed(1)
    X = tf.placeholder(tf.float32, shape=[None, INDIM], name="GraphInput")
    Y = tf.placeholder(tf.float32, shape=[None, OUTDIM], name="GraphTarget")
    LR = tf.placeholder(tf.float32, shape=[], name="LearningRate")
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
    COST = tf.reduce_mean(tf.square(hypothesis - Y), name="Cost")
    OPTIMIZER = tf.train.GradientDescentOptimizer(LR).minimize(COST)

# ---------------------------------
# Graph.
#
for T in TIME_LIST[RUN_TIME]:
    T_W_PATH = TEST_NETWORK_PATH.replace("{{T_TIME}}", 'T{:02d}'.format(T))
    S_W_PATH = SAVE_NETWORK_PATH.replace("{{T_TIME}}", 'T{:02d}'.format(T))
    Make_folder(T_W_PATH), Make_folder(S_W_PATH)

    L_DATA = pd.read_csv(INPUT_PATH + "T{:02d}_LEARN.txt".format(T), delimiter="\t")
    V_DATA = pd.read_csv(INPUT_PATH + "T{:02d}_VALI.txt".format(T), delimiter="\t")
    Learn_x, Learn_y, _ = Data_Split(L_DATA)
    Vali_x, Vali_y, _ = Data_Split(V_DATA)

    # Time information #
    Start_time, Today = time.time(), d.datetime.today()
    Year, Month, Day, Hour, Minute = Today.year, Today.month, Today.day, Today.hour, Today.minute

    # Log file directory & name setting #
    log = open(LOG_PATH + "logfile_T{:02d}.txt".format(T), "w")
    PrintLog = open(LOG_PATH + "PrintLog_T{:02d}.txt".format(T), "w")
    log.write("Start_Time:{:4d}{:02d}{:02d}_{:02d}:{:02d}".format(Year, Month, Day, Hour, Minute) + "\n")
    log.write("COUNT" + "\t" + "LR" + "\t" + "STEP" + "\t" + "EPOCH" + "\t" + "OFFSET" + "\t" +
              "LEAN_COST" + "\t" + "VALI_COST" + "\t" + "MIN_COST" + "\n")

    # Learning rate scheduling #
    print("Start_Time:{:4d}{:02d}{:02d}_{:02d}:{:02d}".format(Year, Month, Day, Hour, Minute))
    COUNT_COST = []
    for COUNT in range(COUNTS):
        VALI_LOSS, OFFSETS = [], []
        ESL = 250 * (COUNT + 1) # Early Stop Length
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=ESL+1)
        ES = EarlyStopping(patience=ESL, verbose=1) # Early Stopping
        epoch = 0
        LEARNING_RATE = 0.09 * 10 ** (-1 * COUNT)
        print('Count:{} // Learning Rate:{:.8f}'.format(COUNT, LEARNING_RATE), file=PrintLog)

        # Restore in LRS process #
        if COUNT > 0:
            saver.restore(sess, T_W_PATH + 'save0')
            print("Resotre weight file : 'save0', T{:02d}".format(T))

        for step in range(MAX_STEP):
            offset = (step * batch_size) % (len(L_DATA) - batch_size)
            OFFSETS.append(offset)
            total_batch = int(len(L_DATA) / batch_size)
            COST_SUM, cru_batch = 0, 0

            for k in range(total_batch):
                batch_x, batch_y = Nextbatch(xdata=Learn_x, ydata=Learn_y,
                                             xcol_num=INDIM, ycol_num=OUTDIM,
                                             size=batch_size, cru=cru_batch, length=len(L_DATA), random=False)
                Cost_Val, _ = sess.run([COST, OPTIMIZER], feed_dict={X: batch_x, Y: batch_y, LR: LEARNING_RATE})
                COST_SUM += Cost_Val
                cru_batch += batch_size

            if step != 0 and OFFSETS[step-1] > OFFSETS[step]:
                VALI_CONC = sess.run(hypothesis, feed_dict={X: Vali_x})
                VALI_df = pd.DataFrame([], columns=['AI', 'Target'])
                CONC_list = np.zeros(shape=len(V_DATA), dtype=np.float64).tolist()

                for i in range(len(V_DATA)):
                    CONC_list[i] = VALI_CONC[i][0]

                VALI_df['AI'], VALI_df['Target'] = CONC_list, Vali_y
                VALI_df['SE'] = (VALI_df['AI'] - VALI_df['Target'])**2
                VALI_COST = VALI_df['SE'].mean()
                VALI_LOSS.append(VALI_COST)

                if epoch % 5 == 0:
                    print("COUNT:{:d} // LR:{:f} // STEP:{:d} // EPOCH:{:d} // OFFSET:{:d} // LEARN_COST:{:.4f} // VALI_COST:{:.4f} // MIN_COST:{:.4f}"
                          .format(COUNT, LEARNING_RATE, step, epoch, offset, COST_SUM, VALI_COST, np.min(VALI_LOSS)), file=PrintLog)
                    print("COUNT:{:d} // LR:{:f} // STEP:{:d} // EPOCH:{:d} // OFFSET:{:d} // LEARN_COST:{:.4f} // VALI_COST:{:.4f} // MIN_COST:{:.4f}"
                          .format(COUNT, LEARNING_RATE, step, epoch, offset, COST_SUM, VALI_COST, np.min(VALI_LOSS)))

                    log.write("{:d}".format(COUNT) + "\t" + "{:f}".format(LEARNING_RATE) + "\t" +
                              "{:d}".format(step) + "\t" + "{:d}".format(epoch) + "\t" + "{:.4f}".format(offset) + "\t" +
                              "{:.4f}".format(COST_SUM) + "\t" + "{:.4f}".format(VALI_COST) + "\t" + "{:.4f}".format(np.min(VALI_LOSS)) + "\n")

                saver.save(sess, S_W_PATH, global_step=epoch)
                epoch += 1

                if ES.validate(VALI_COST, FILE=PrintLog):
                    print('\n', file=PrintLog)
                    print('Training process is stopped early....', file=PrintLog)
                    break


        saved_epoch = epoch - (ESL - 1)
        COUNT_COST.append(VALI_LOSS[saved_epoch])

        if COUNT == 0:
            file_save(saved_epoch, S_W_PATH, T_W_PATH)
            print('%i count file save, Test loss: %4f' %(COUNT, VALI_LOSS[saved_epoch]))

        if COUNT != 0:
            if COUNT_COST[COUNT] < COUNT_COST[COUNT - 1]:
                file_save(saved_epoch, S_W_PATH, T_W_PATH)
                print('%i count file save, Test loss: %4f' %(COUNT, VALI_LOSS[saved_epoch]))

            else:
                file_delete(S_W_PATH)
                print("%i count don't save, Process termination ..." % COUNT)
                break
    End_time = time.time()
    print("\n")
    log.close()
