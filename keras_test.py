import numpy as np
import pandas
import random
import keras
import sklearn
import os

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.metrics import categorical_accuracy
from sklearn.preprocessing import LabelEncoder

def load_train_data(inputPath, outputPath):
    # Pre-processed Data
    dataFrame = pandas.read_csv(inputPath, delimiter=',', header = None)
    inputDataSet = dataFrame.values
    dataFrame = pandas.read_csv(outputPath, delimiter=',', header = None)
    outputDataSet = dataFrame.values
    
    numInput = len(inputDataSet[0])
    numOutput = len(outputDataSet[0])

    return inputDataSet, outputDataSet, numInput, numOutput
    
def split_data(inputDataSet, outputDataSet, num_train = 850000):
    # Split data into 2 parts : train / test
    #num_train = 850000
    x_train = inputDataSet[:num_train]
    y_train = outputDataSet[:num_train]
    x_test = inputDataSet[num_train:]
    y_test = outputDataSet[num_train:]
    
    print(len(inputDataSet)) 
    print(len(outputDataSet))
    print(len(inputDataSet[0])) # 4
    print(len(outputDataSet[0])) # 1

    return x_train, y_train, x_test, y_test

# DNN Model < input: 20774 -> hidden1: 512 -> hidden2: 256 -> hidden3: 128  -> hidden4 : 64 -> output: 6 >
def baselineModel(numInput, numOutput):
    model = Sequential()
    model.add(Dense(1024, input_dim=numInput))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(numOutput))
    model.add(Activation('sigmoid'))
    adam = Adam(lr=1e-4)
    #sgd = SGD(lr=1e-4)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def load_or_create_model(numInput, numOutput, modelFile):
    #modelFile = 'DNN_Model.h5'
    #if os.path.isfile(modelFile):
    #    myModel = load_model(modelFile)
    #else:
    #    myModel = baselineModel(numInput, numOutput)
    index = 0
    if not os.path.isfile(modelFile + '.' + str(index)):
        myModel = baselineModel(numInput, numOutput)
    else:
        while os.path.isfile(modelFile + '.' + str(index+1)):
            index += 1
        myModel = load_model(modelFile + '.' + str(index))

    #myModel = baselineModel()
    print(myModel.summary())

    return myModel, index

def train_and_save_model(myModel, inputDataSet, outputDataSet, modelFile):
    myModel.fit(inputDataSet, outputDataSet, epochs=5, batch_size=100, validation_split=0.1, verbose=1)
    if modelFile != None:
        myModel.save(modelFile)

    return myModel

def load_test_data(testInput, testOutput, frac = 1):
    dataFrame = pandas.read_csv(testInput, delimiter=',', header = None)
    inputDataSet = dataFrame.values
    inputDataSet = inputDataSet[:(len(inputDataSet) * frac)]
    dataFrame = pandas.read_csv(testOutput, delimiter=',', header = None)
    outputDataSet = dataFrame.values
    outputDataSet = outputDataSet[:(len(outputDataSet) * frac)]

    return inputDataSet, outputDataSet

def evaluate_model(myModel, inputDataSet, outputDataSet):
    prediction_result = myModel.predict(inputDataSet, verbose=1)
    score = myModel.evaluate(inputDataSet, outputDataSet, verbose=1)
    print(outputDataSet[0:20])
    print(prediction_result[0:20])

    # Count how many 
    correctCount = 0
    falseNeg = 0
    falsePos = 0
    numberOfOne = 0
    numberOfZero = 0
    for i in range(len(prediction_result)):
        test = prediction_result[i]
        cor = outputDataSet[i]
        if cor == 1:
            numberOfOne += 1
        else:
            numberOfZero += 1
    
        if abs(cor - test) < 0.5:
            correctCount += 1
    
        if cor == 1 and test < 0.5:
            falseNeg += 1
    
        if cor == 0 and test >= 0.5: 
            falsePos += 1
    
    print('Loss : %.4f' % score[0])
    print('Accuracy : %.4f' % score[1])
    print('Correct : %d / %d' % (correctCount, len(prediction_result)))
    print('False Negative : %d' % falseNeg)
    print('False Positive : %d' % falsePos)
    print('Number of Ones : %d' % numberOfOne)
    print('Number of Zeros : %d' % numberOfZero)

    return falseNeg

if __name__ == '__main__':
    inputPath = './input_gen.csv'
    outputPath = './output_gen.csv'
    inputDataSet, outputDataSet, numInput, numOutput = load_train_data(inputPath, outputPath)
    testInput = './input_real.csv'
    testOutput = './output_real.csv'
    frac = 0.05
    inputTestDataSet, outputTestDataSet = load_test_data(testInput, testOutput, frac)
    modelFile = 'DNN_Model.h5'
    omitting_steps = 0 # make a checkpoint in omitting_steps+1
    model, index = load_or_create_model(numInput, numOutput, modelFile)
    while True:
        for i in range(0, omitting_steps):
            train_and_save_model(model, inputTestDataSet, outputTestDataSet, None)
            falseNeg = evaluate_model(model, inputDataSet, outputDataSet)
        index += 1
        train_and_save_model(model, inputTestDataSet, outputTestDataSet, modelFile + '.' + str(index))
        falseNeg = evaluate_model(model, inputDataSet, outputDataSet)
        with open("./falseNeg_history.csv") as myfile:
            myfile.write("{},{}\n".format(index, falseNeg))
