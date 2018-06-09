import pandas
import sklearn
import os

data_path = '/home/woori4829/talkingdata-adtracking-fraud-detection/train_sample.csv'
#data_path = '/home/woori4829/talkingdata-adtracking-fraud-detection/train.csv' 

# Read from file, Eliminate first row (with header = 0)
dataFrame = pandas.read_csv(data_path, delimiter=',', header = 0)
dataSet = dataFrame.values

# Shuffle all the users' data
numData = len(dataSet)

# To store unique elems for each column
uniqueElems = []

# 0: ip, 1: app, 2: device, 3: os, 4: channel, 5: click_time, 6: attribute_time, 7: is_attributed
for i in range(len(dataSet[0])):
    uniqueElems.append(list(set(dataSet[:,i])))
    print(len(uniqueElems[i]))

ips = uniqueElems[0]
ipDict = {}
for i in range(len(ips)):
    ipDict[ips[i]] = i

apps = uniqueElems[1]
appDict = {}
for i in range(len(apps)):
    appDict[apps[i]] = i

devices = uniqueElems[2]
deviceDict = {}
for i in range(len(devices)):
    deviceDict[devices[i]] = i

oss = uniqueElems[3]
osDict = {}
for i in range(len(oss)):
    osDict[oss[i]] = i

channels = uniqueElems[4]
channelDict = {}
for i in range(len(channels)):
    channelDict[channels[i]] = i

trainInput = []
trainOutput = dataSet[:,7]

# Process all data
for data in dataSet:
    ip = data[0]
    app = data[1]
    dev = data[2]
    os_ = data[3]
    chan = data[4]
    clickT = data[5]
    isattr = data[7]

    # TODO : split click time and get Hour & Min from the data
    tokens = clickT.split(" ")
    times = tokens[1].split(":")

    inputList = [ipDict[ip], appDict[app], deviceDict[dev], osDict[os_], channelDict[chan], times[0], times[1]]
    trainInput.append(inputList)

inputPath = 'input.csv'
outputPath = 'output.csv'
#inputPath = 'input_real.csv'
#outputPath = 'output_real.csv'

# Check whether the file exists or not
if os.path.isfile(inputPath):
    os.remove(inputPath)
if os.path.isfile(outputPath):
    os.remove(outputPath)

inputFile = open(inputPath,'w')
outputFile = open(outputPath,'w')

# Write to file
for i in range(len(trainInput)):
    intext = ''
    intext += ','.join(map(str,trainInput[i]))
    intext += '\n'
    outtext = ''
    outtext += str(trainOutput[i])
    outtext += '\n'
    inputFile.write(intext)
    outputFile.write(outtext)

inputFile.close()
outputFile.close()
