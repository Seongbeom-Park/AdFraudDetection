import pandas
import sklearn
import os

#data_path = '/home/woori4829/talkingdata-adtracking-fraud-detection/train_sample.csv'
negDataPath = '/home/woori4829/AdFraudDetection/large_negatives.csv' 
posDataPath = '/home/woori4829/AdFraudDetection/large_positives.csv' 

# Read from file, Eliminate first row (with header = 0)
negDataFrame = pandas.read_csv(negDataPath, delimiter=',', header = None)
negDataSet = negDataFrame.values
posDataFrame = pandas.read_csv(posDataPath, delimiter=',', header = None)
posDataSet = posDataFrame.values

# Shuffle all the users' data
numpPosData = len(posDataSet)

wholeData = []
# Process all data
negDataSet = sklearn.utils.shuffle(negDataSet)
for i in range(len(posDataSet)):
    wholeData.append(negDataSet[i])

for data in posDataSet:
    wholeData.append(data)

outputPath = 'dataGen.csv'

for i in range(30):
    wholeData = sklearn.utils.shuffle(wholeData)

# Check whether the file exists or not
if os.path.isfile(outputPath):
    os.remove(outputPath)

outputFile = open(outputPath,'w')

# Write to file
for i in range(len(wholeData)):
    intext = ''
    intext += ','.join(map(str,wholeData[i]))
    intext += '\n'
    outputFile.write(intext)

outputFile.close()
