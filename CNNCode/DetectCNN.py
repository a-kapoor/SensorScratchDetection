#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os

from Config import *


# In[2]:


import os


def folder_creat(name, directory):
    os.chdir(directory)
    fileli = os.listdir()
    if name in fileli:
        print(f'Folder "{name}" exist!')
    else:
        os.mkdir(name)
        print(f'Folder "{name}" succesfully created!')
        


folder_creat(outputdirname, './')


# In[3]:


from PIL import Image,ImageOps
import numpy as np
import glob

def filetoarray(imagenum,inputfolder='./canny'):
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.preprocessing.image import array_to_img
    img = load_img(inputfolder+'/sensorphoto-'+str(imagenum)+'-1.BMP.png',color_mode="grayscale",target_size=targetinputshape)
    img_array = img_to_array(img)
    return img_array


# In[4]:


import os

traindir = os.listdir(inputfolderTrain)
if len(traindir) == 0:
    print("Empty Training directory")
else:
    print("Found Training images")
    trdf=pd.read_csv(ScratchFileTrain)
    trdf['ImageLoc'] = trdf.ImageLoc.astype(int)
    trdf['array'] = trdf.apply(lambda row: filetoarray(row.ImageLoc,inputfolderTrain), axis=1)
    trdf['Scratched'] = 0
    trdf.loc[trdf['Target'] ==11,'Scratched'] = 1
    trdf.loc[trdf['Target'] ==1,'Scratched'] = 1
    print('Scratched Images in Training dataset = '+str(len(trdf.query('Scratched==1'))))
    print('Clean Images in Training dataset = '+str(len(trdf.query('Scratched==0'))))
    X_train=np.array(list(trdf["array"]))
    Y_train=trdf["Scratched"]
    Y_train = to_categorical(Y_train)
    print("Train shape: "+str(X_train.shape))
    
testdir = os.listdir(inputfolderTest)
if len(testdir) == 0:
    print("Empty Testing directory")
else:
    print("Found Testing images")
    tedf=pd.read_csv(ScratchFileTest)
    tedf['ImageLoc'] = tedf.ImageLoc.astype(int)
    tedf['array'] = tedf.apply(lambda row: filetoarray(row.ImageLoc,inputfolderTest), axis=1)
    tedf['Scratched'] = 0
    tedf.loc[tedf['Target'] ==11,'Scratched'] = 1
    tedf.loc[tedf['Target'] ==1,'Scratched'] = 1
    print('Scratched Images in Testing dataset = '+str(len(tedf.query('Scratched==1'))))
    print('Clean Images in Testing dataset = '+str(len(tedf.query('Scratched==0'))))
    X_test=np.array(list(tedf["array"]))
    Y_test=tedf["Scratched"]
    Y_test = to_categorical(Y_test)
    print("Test shape: "+str(X_test.shape))


evaldir = os.listdir(inputfolderEvaluate)
if len(evaldir) == 0:
    print("Empty Eval directory, will not evaluate")
else:
    print("Found Eval images")
    evdf=pd.read_csv(ScratchFileEvaluate)
    evdf['ImageLoc'] = evdf.ImageLoc.astype(int)
    evdf['array'] = evdf.apply(lambda row: filetoarray(row.ImageLoc,inputfolderEvaluate), axis=1)
    evdf['Scratched'] = 0
    evdf.loc[evdf['Target'] ==11,'Scratched'] = 1
    evdf.loc[evdf['Target'] ==1,'Scratched'] = 1
    print('Scratched Images in Evaluate dataset = '+str(len(evdf.query('Scratched==1'))))
    print('Clean Images in Evaluate dataset = '+str(len(evdf.query('Scratched==0'))))
    X_eval=np.array(list(evdf["array"]))
    Y_eval=evdf["Scratched"]
    Y_eval = to_categorical(Y_eval)
    print("Eval shape: "+str(X_eval.shape))


# In[5]:


from tensorflow.keras import optimizers
from tensorflow.keras.losses import categorical_crossentropy
model.compile(loss=categorical_crossentropy,
              optimizer=optimizers.Adam(lr=learningrate),
              metrics=['accuracy'])

trainhistory=model.fit(X_train, Y_train,batch_size=batch_size,epochs=epochs,verbose=1)


# In[6]:


if len(traindir) != 0:
    score = model.evaluate(X_train, Y_train, verbose=0)
    Y_train_pred = model.predict_classes(X_train)
    print('Train loss:', score[0])
    print('Train accuracy:', score[1])


# In[7]:


if len(testdir) != 0:
    score = model.evaluate(X_test, Y_test, verbose=0)
    Y_test_pred = model.predict_classes(X_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


# In[8]:


if len(evaldir) != 0:
    score = model.evaluate(X_eval, Y_eval, verbose=0)
    Y_eval_pred = model.predict_classes(X_eval)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


# In[10]:


from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(np.argmax(Y_test,axis=-1),  Y_test_pred)
auc = metrics.roc_auc_score(np.argmax(Y_test,axis=-1),  Y_test_pred)
plt.plot(fpr,tpr,label="test, auc="+str(auc))
plt.legend(loc=4)
plt.savefig(outputdirname+"/Test-ROC.png")


# In[11]:


fpr, tpr, _ = metrics.roc_curve(np.argmax(Y_train,axis=-1),  Y_train_pred)
auc = metrics.roc_auc_score(np.argmax(Y_train,axis=-1),  Y_train_pred)
plt.plot(fpr,tpr,label="train, auc="+str(auc))
plt.legend(loc=4)
plt.savefig(outputdirname+"/Train-ROC.png")


# In[12]:


if len(evaldir) != 0:
    print("True Eval")
    print(np.argmax(Y_train, axis=-1))
    print("Predicted Eval")
    print(Y_train_pred)
else:
    print("No eval images, so won't evaluate")


# In[ ]:


print("Find plots in directory: "+outputdirname)

