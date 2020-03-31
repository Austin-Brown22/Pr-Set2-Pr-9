from sklearn.neural_network import MLPClassifier
from sklearn import svm
import csv

from warnings import filterwarnings
filterwarnings('ignore')

#list of training points, each item is a list of floats with a char at the end that tells classification
# each point has 34 float values
trainingData = []
# the correct classification of each coresponding training point
trainingClass = []
#trainingData[i] has classification trainingClass[i]

pathToTrainingData= 'C:/Users/austi/Documents/AI WORK/Poblem set 2/Problem 9/'
with open(pathToTrainingData+'ionosphere.csv') as csvFile:
    reader = csv.reader(csvFile, delimiter= ',')
    for row in reader:
        trainingData.append(row[:-1])
        trainingClass.append(row[-1])

print(type(trainingData[0][0]))
for i in range(len(trainingData)):
    for j in range(len(trainingData[0])):
        trainingData[i][j] = float(trainingData[i][j])


# divide the data into 10 groups for cross validation 
numExamples = len(trainingClass)
numInGoup = numExamples // 10
# list of groups, each goup is a list of data points
groups = []
groupClass= []
for i in range(10):
    groups.append(trainingData[i*35:(i+1)*35])
    groupClass.append(trainingClass[i*35:(i+1)*35])

groups[9].append(trainingData[350])
groupClass[9].append(trainingClass[350])
lineargroupEval = []

# run all 10 iterations of traing and testing to generate a list of evaluation scores
print("Linar SVM")
for i in range(10):

    holdGroup = groups[i]
    holdGoupCla = groupClass[i]
    trainPoints = []
    trainClass = []
    for j in range(10):
        if j != i:
            for point in groups[j]:
                trainPoints.append(point)
            for cass in groupClass[j]:
                trainClass.append(cass)


    #print(len(trainPoints))
    #print(trainPoints)
    #print(len(trainClass))
    #  print(trainClass)

    machine = svm.SVC(kernel="linear")
    machine.fit(trainPoints, trainClass)

    predictions = machine.predict(holdGroup)
    cnt = 0
    for guess in predictions:
        if(guess != trainingClass[cnt]):
            cnt += 1
    failRate = cnt/len(predictions)*100
    print("Trial",i,"had a", str(failRate)+"%", "failure rate")
    lineargroupEval.append(failRate)

# run all 10 iterations of traing and testing to generate a list of evaluation scores
polygroupEval = []
print("polynomial SVM")
for i in range(10):

    holdGroup = groups[i]
    holdGoupCla = groupClass[i]
    trainPoints = []
    trainClass = []
    for j in range(10):
        if j != i:
            for point in groups[j]:
                trainPoints.append(point)
            for cass in groupClass[j]:
                trainClass.append(cass)


    #print(len(trainPoints))
    #print(trainPoints)
    #print(len(trainClass))
    #  print(trainClass)

    machine = svm.SVC(kernel="poly", degree=3)
    machine.fit(trainPoints, trainClass)

    predictions = machine.predict(holdGroup)
    cnt = 0
    for guess in predictions:
        if(guess != trainingClass[cnt]):
            cnt += 1
    failRate = cnt/len(predictions)*100
    print("Trial",i,"had a", str(failRate)+"%", "failure rate")
    polygroupEval.append(failRate)

# NEUAL NETWORK

# run all 10 iterations of traing and testing to generate a list of evaluation scores
Neural1groupEval = []
print("Nural Net relu")
for i in range(10):

    holdGroup = groups[i]
    holdGoupCla = groupClass[i]
    trainPoints = []
    trainClass = []
    for j in range(10):
        if j != i:
            for point in groups[j]:
                trainPoints.append(point)
            for cass in groupClass[j]:
                trainClass.append(cass)


    #print(len(trainPoints))
    #print(trainPoints)
    #print(len(trainClass))
    #  print(trainClass)

    machine = MLPClassifier(hidden_layer_sizes= (1))
    machine.fit(trainPoints, trainClass)

    predictions = machine.predict(holdGroup)
    cnt = 0
    for guess in predictions:
        if(guess != trainingClass[cnt]):
            cnt += 1
    failRate = cnt/len(predictions)*100
    print("Trial",i,"had a", str(failRate)+"%", "failure rate")
    polygroupEval.append(failRate)


# run all 10 iterations of traing and testing to generate a list of evaluation scores
Neural1groupEval = []
print("Nural Net Logistic")
for i in range(10):

    holdGroup = groups[i]
    holdGoupCla = groupClass[i]
    trainPoints = []
    trainClass = []
    for j in range(10):
        if j != i:
            for point in groups[j]:
                trainPoints.append(point)
            for cass in groupClass[j]:
                trainClass.append(cass)


    #print(len(trainPoints))
    #print(trainPoints)
    #print(len(trainClass))
    #  print(trainClass)

    machine = MLPClassifier(hidden_layer_sizes= (17), activation="logistic")
    machine.fit(trainPoints, trainClass)

    #print(machine.n_layers_)

    predictions = machine.predict(holdGroup)
    cnt = 0
    for guess in predictions:
        if(guess != trainingClass[cnt]):
            cnt += 1
    failRate = cnt/len(predictions)*100
    print("Trial",i,"had a", str(failRate)+"%", "failure rate")
    polygroupEval.append(failRate)