# GMD
Geoscientific Model Development


The repository provides the code used in the paper submitted to Geoscientific Model Development (GMD).\
The title of the paper is "Development of a deep natural network for presenting 6-hour average PM2.5 concentrations up to two subsequential days using diverse training data".\
This study aims to develop a deep neural network (DNN) model as an artificial neural network (ANN) for the prediction of 6-hour average fine particulate matter (PM2.5) concentrations for a three-day period—the day of prediction (D+0), one day after prediction (D+1) and two days after prediction (D+2)—using observation data and forecast data obtained via numerical models.\
In addition, three experiments were performed to examine the effects of the training-data configuration on the prediction performance of DNN model. The DNN-OBS model used the observation data as the sole training data, the DNN-OPM model used both observation and weather forecast data as the training data, and the DNN-ALL model used the observation data, weather forecast data, and PM2.5 concentration prediction data as the training data.\
Therefore, this repository provides code and data for performing the three DNN models developed in this paper.\
The computer language used in the development is python, and the DNN model was constructed using tensorflow.\


Code and folder describtion\
./DNN-ALL/ :: It contains codes and data related to the DNN-ALL model.\
-> ./DNN-ALL/INPUT/ : This folder contains training, validation, and test data used in the DNN-ALL model.\
-> ./DNN-ALL/LOG/ : In this folder, there are log files in which the training process of the DNN-ALL model is recorded.\
-> ./DNN-ALL/T_RESULT/ : This folder stores the test results of the DNN-ALL model.\
-> ./DNN-ALL/Module/ : The module code used in the DNN-ALL model is included.\
-> ./DNN-ALL/Module/config.py : This script defines the variables required to perform the DNN-ALL model.\
-> ./DNN-ALL/Module/utile.py : This script has a custom function required to perform the DNN-ALL model.\
-> ./DNN-ALL/Module/PostModule.py : This script has a custom function required to perform post-processing of the DNN-ALL model.\
-> ./DNN-ALL/TEST_NETWORK/ : This folder stores a network file that is a training result of the DNN-ALL model.\
-> ./DNN-ALL/TrainingDNN.py : It is a training script for the DNN-ALL model.\
-> ./DNN-ALL/PredictDNN.py : It is a script that produces prediction results by applying network to test data.\
-> ./DNN-ALL/Postprocessing.py : It is a script that performs statistical and AQI evaluations.

./DNN-OBS/ :: It contains codes and data related to the DNN-OBS model. The folder package configuration is the same as the DNN-ALL folder package.

./DNN-OPM/ :: It contains codes and data related to the DNN-OPM model. The folder package configuration is the same as the DNN-ALL folder package.

./Module/ :: The module code used in DataPreprocessing.py is included.\
-> ./Module/config.py : This script defines the variables required to perform DataPreprocessing.py.\
-> ./Module/utile.py : This script has a custom function required to perform DataPreprocessing.py.

./RawData/ :: It contains the original data used in this paper.\
-> ./RawData/SU_f_09 : It is prediction data of CMAQ and WRF models in Seoul.\
-> ./RawData/SU_obs_09 : It is observation data in Seoul.

./DataPreprocessing.py : This script processes the raw data to generate training, validation and test data of DNN-OBS, DNN-OPM, and DNN-ALL. In addition, membership function data is generated.
