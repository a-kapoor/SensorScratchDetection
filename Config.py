import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

##### Start here

DataFile='modified.txt'
OutputDir='Output'

testSize=0.2

modelDNN_DNN_1=Sequential()
modelDNN_DNN_1.add(Dense(5000, kernel_initializer='glorot_normal', activation='relu', input_dim=5427))
modelDNN_DNN_1.add(Dense(2500, kernel_initializer='glorot_normal', activation='relu'))
modelDNN_DNN_1.add(Dropout(0.2))
modelDNN_DNN_1.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))

DNNDict={'epochs':100, 'batchsize':10, 'lr':0.001, 'model':modelDNN_DNN_1,'earlyStop':True}