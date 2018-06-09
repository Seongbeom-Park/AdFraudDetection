import pandas
import sklearn
import os

#data_path = '/home/woori4829/talkingdata-adtracking-fraud-detection/train_sample.csv'
data_path = '/home/woori4829/talkingdata-adtracking-fraud-detection/train.csv' 

# Read from file, Eliminate first row (with header = 0)
dataFrame = pandas.read_csv(data_path, delimiter=',', header = 0)
dataSet = dataFrame.values

# Shuffle all the users' data
numData = len(dataSet)

negatives = []
positives = []
# Process all data
for data in dataSet:
    isattr = data[7]

    if isattr == 1:
        positives.append(data)
    else:
        negatives.append(data)

#negPath = 'small_negatives.csv'
#posPath = 'small_positives.csv'
negPath = 'large_negatives.csv'
posPath = 'large_positives.csv'

# Check whether the file exists or not
if os.path.isfile(negPath):
    os.remove(negPath)
if os.path.isfile(posPath):
    os.remove(posPath)

negFile = open(negPath,'w')
posFile = open(posPath,'w')

# Write to file
for i in range(len(negatives)):
    intext = ''
    intext += ','.join(map(str,negatives[i]))
    intext += '\n'
    negFile.write(intext)

for i in range(len(positives)):
    intext = ''
    intext += ','.join(map(str,positives[i]))
    intext += '\n'
    posFile.write(intext)

negFile.close()
posFile.close()
