from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

#------------------------------------------ 
outputdirname='results'
#------------------------------------------

ScratchFileTrain="data/canny_train_targets.csv"
inputfolderTrain='data//canny_train/'

#------------------------------------------

ScratchFileTest="data/canny_test_targets.csv"
inputfolderTest='data/canny_test/'

#------------------------------------------

ScratchFileEvaluate="data/canny_evaluate_targets.csv"
inputfolderEvaluate='data/canny_evaluate/'

#------------------------------------------

targetinputshape=(100,100) #Can also keep original size if you wish

#------------------------------------------
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(*targetinputshape,1)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(5, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
#------------------------------------------
learningrate=0.01
batch_size=64
epochs=50
