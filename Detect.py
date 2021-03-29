#!/usr/bin/env python
# coding: utf-8

# ![image](Sensorpic.png)

# In[1]:


RandomState=42 #Choose same number for reproducibility


# In[18]:


from Tools.PlotTools import *
import os
import Config as Conf
os.system("mkdir -p " + Conf.OutputDir)


# In[3]:


import pandas as pd
df_final = pd.read_csv(Conf.DataFile, header=None)
print("Total photos : "+str(len(df_final)))
df_final.head()


# In[4]:


#Fill all empty (NaN) columns with zeros
df_final.fillna(0,inplace=True)

#Rather than 0,1,2,3,4, I want to name them F1,F2,F3 etc.
df_final.columns=["F"+str(i) for i in range(1, 5429+1)]

#Check first few rows
df_final.head()


# In[5]:


#Get all the column names
Col=df_final.columns.to_list()

#Features except name and label
features=Col
features.remove("F1") #Remove image name
features.remove("F2") #Remove Label name
print("Features = "+str(len(features)))

#Create label
df_final['Scratched'] = 0
df_final.loc[df_final['F2'] ==11,'Scratched'] = 1
df_final.loc[df_final['F2'] ==1,'Scratched'] = 1

cat='Scratched'

print("Background = "+str(len(df_final.query(cat+"==0"))))
print("Signal = "+str(len(df_final.query(cat+"==1"))))

df_final['Wt'] = 1

df_final.head()


# In[6]:


testSize=Conf.testSize #70%
from sklearn.model_selection import train_test_split
TrainIndices, TestIndices = train_test_split(df_final.index.values.tolist(), test_size=testSize, random_state=RandomState, shuffle=True)


# In[7]:


df_final.loc[TrainIndices,'Dataset'] = "Train"
df_final.loc[TestIndices,'Dataset'] = "Test"

df_final.loc[TrainIndices,'TrainDataset'] = 1
df_final.loc[TestIndices,'TrainDataset'] = 0


# In[8]:


import numpy as np
X_train = np.asarray(df_final.loc[TrainIndices,features])
Y_train = np.asarray(df_final.loc[TrainIndices,cat])

X_test = np.asarray(df_final.loc[TestIndices,features])
Y_test = np.asarray(df_final.loc[TestIndices,cat])

RunDNN=True


# In[9]:


if RunDNN:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    DNNDict=Conf.DNNDict
    modelDNN=DNNDict['model']
    modelDNN.compile(loss='binary_crossentropy', optimizer=Adam(lr=DNNDict['lr']), metrics=['accuracy',])
    
    if DNNDict['earlyStop']:
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        train_history = modelDNN.fit(X_train,Y_train,epochs=DNNDict['epochs'],batch_size=DNNDict['batchsize'],validation_data=(X_test,Y_test),
                                 verbose=1,callbacks=[es])
    else:
        train_history = modelDNN.fit(X_train,Y_train,epochs=DNNDict['epochs'],batch_size=DNNDict['batchsize'],validation_data=(X_test,Y_test),
                                 verbose=1)
    #modelDNN.save(Conf.OutputDirName+"/"+MVA+"/"+MVA+"_"+"modelDNN.h5")
    df_final.loc[TrainIndices,"DNN"+"_pred"]=modelDNN.predict(X_train)
    df_final.loc[TestIndices,"DNN"+"_pred"]=modelDNN.predict(X_test)


# In[11]:


if RunDNN:
    prGreen("Plotting output response for DNN")
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    plot_mva(df_final.query('TrainDataset==1'),"DNN_pred",bins=np.linspace(0, 1, 51),cat=cat,ax=axes,sample='train',ls='dashed',logscale=False)
    plt.savefig(Conf.OutputDir+"/TrainMVA.png")
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    plot_mva(df_final.query('TrainDataset==0'),"DNN_pred",bins=np.linspace(0, 1, 51),cat=cat,ax=axes,sample='test',ls='dashed',logscale=False)
    plt.savefig(Conf.OutputDir+"/TestMVA.png")


# In[12]:


df_final.loc[df_final["DNN_pred"]>0.5,"predlabel"]=1
df_final.loc[df_final["DNN_pred"]<0.5,"predlabel"]=0


# In[19]:


import seaborn as sns
confusion_matrix = pd.crosstab(df_final.query('TrainDataset==1')[cat], df_final.query('TrainDataset==1')["predlabel"], rownames=['Actual'], colnames=['Predicted'])
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
sns_plot=sns.heatmap(confusion_matrix,cmap="YlGnBu", annot=True, cbar=False,fmt='g',ax=axes)
plt.savefig(Conf.OutputDir+"/confusion_matrix_train.png")


# In[21]:


import seaborn as sns
confusion_matrix = pd.crosstab(df_final.query('TrainDataset==0')[cat], df_final.query('TrainDataset==0')["predlabel"], rownames=['Actual'], colnames=['Predicted'])
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
sns_plot=sns.heatmap(confusion_matrix,cmap="YlGnBu", annot=True, cbar=False,fmt='g',ax=axes)
plt.savefig(Conf.OutputDir+"/confusion_matrix_test.png")


# In[ ]:




