import numpy as np
import pandas
import random
import keras
import sklearn

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.metrics import categorical_accuracy
from sklearn.preprocessing import LabelEncoder

# Pre-processed Data
inputPath = './input.csv'
outputPath = './output.csv'

dataFrame = pandas.read_csv(inputPath, delimiter=',', header = None)
inputDataSet = dataFrame.values
dataFrame = pandas.read_csv(outputPath, delimiter=',', header = None)
outputDataSet = dataFrame.values

numInput = len(inputDataSet[0])
numOutput = len(outputDataSet[0])

print(len(inputDataSet)) # 100000
print(len(outputDataSet)) # 100000
print(len(inputDataSet[0])) # 4
print(len(outputDataSet[0])) # 1

# DNN Model < input: 20774 -> hidden1: 512 -> hidden2: 256 -> hidden3: 128  -> hidden4 : 64 -> output: 6 >
def baselineModel():
    model = Sequential()
    model.add(Dense(4096, input_dim=numInput))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(numOutput))
    model.add(Activation('sigmoid'))
    adam = Adam(lr=1e-4)
    #sgd = SGD(lr=1e-4)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

# Split data into 2 parts : train / test
num_train = 90000
x_train = inputDataSet[:num_train]
y_train = outputDataSet[:num_train]
x_test = inputDataSet[num_train:]
y_test = outputDataSet[num_train:]

print(x_train[0:10])
print(y_train[0:10])
print(x_test[0:10])
print(y_test[0:10])
myModel = baselineModel()
print(myModel.summary())
myModel.fit(x_train, y_train, epochs=20, batch_size=100, validation_split=0.1, verbose=1)

prediction_result = myModel.predict(x_test, verbose=1)
score = myModel.evaluate(x_test, y_test, verbose=1)
print(prediction_result[0:10])

# Count how many 
correctCount = 0
correctOnes = 0
for i in range(len(prediction_result)):
    test = prediction_result[i]
    cor = y_test[i]
    if (test > 0.1 and cor == 1) or (test <= 0.1 and cor == 0):
        correctCount += 1
    if (cor == 1):
        print(test)
        correctOnes += 1

print('Loss : %.4f' % score[0])
print('Accuracy : %.4f' % score[1])
print('Correct : %d / %d' % (correctCount, len(prediction_result)))
print('Correct Ones : %d' % correctOnes)
