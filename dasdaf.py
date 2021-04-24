import numpy as np
import scipy
from scipy.spatial import distance

T = 70000
feature_length = 3
TrainLen = 10000
TestLen = 50000
ValLen = 10000




X_Train = np.zeros((TrainLen, feature_length))
Y_Train = np.zeros((TrainLen, 1))
X_Test = np.zeros((TestLen, feature_length))
Y_Test = np.zeros((TestLen, 1))
X_Val = np.zeros((ValLen, feature_length))
Y_Val = np.zeros((ValLen, 1))


def dist(X,TrainX):
    Dist = np.zeros(len(TrainX))
    for i in range(0, len(TrainX)):
        Dist[i] = np.sqrt(np.sum((X-TrainX[i])**2))
    return Dist
C = 3 #Number of classes

def sort_train_labels_knn(Dist):
    MatrixIndex = []
    for i in range(0, TrainLen):
        MatrixIndex.append((Dist[i], int(i)))
    MatrixIndex.sort()
    a = np.array(MatrixIndex)
    return a

def Probability(sorted_Index, Y, K, Th):
    UD = []
    for i in range(0,K):
        UD.append(Y[int(sorted_Index[i][1])])
    if sum(UD)/K > Th/10:
        return 1
    if sum(UD)/K < -Th/10:
        return -1
    else:
        return 0

pullData = open('AUDCAD1H.txt', 'r').read()
dataArray = pullData.split('\n')
xar= []

for eachLine in dataArray:
    if len(eachLine) >1:
        x,y,z,e,p,w = eachLine.split(",")
        xar.append(float(z))



diff =[]
for l in range(0,T):
    if xar[l+feature_length] - xar[l+feature_length+1] < 0:
        diff.append(1)
    if xar[l+feature_length] - xar[l+feature_length+1] > 0:
        diff.append(-1)
    if xar[feature_length+l] - xar[l+feature_length+1]  == 0:
        diff.append(0)

for i in range(0, TrainLen):
    for l in range(0, feature_length):
        X_Train[i][l] = (xar[i+l] - xar[i+l+1]) / xar[i+l+1]
for i in range(0, TrainLen):
    for l in range(0,1):
        Y_Train[i][l] =  diff[i+l]


for i in range(TrainLen, TrainLen+TestLen):
    for l in range(0, feature_length):
        X_Test[TrainLen - i][l] = (xar[i+l] - xar[i+l+1]) / xar[i+l+1]
for i in range(TrainLen, TrainLen+TestLen):
    for l in range(0,1):
         Y_Test[TrainLen - i][l] =  diff[i+l]


Distance = np.zeros((TrainLen))
Index = np.zeros((TrainLen))

Threshold = 7
K = 11
counter = 0
total =0
mean =0
totalM =[[] for i in range (0, 10)]
counterM =[[] for i in range (0, 10)]
meanM =[[] for i in range (0, 10)]
print(counterM)
counterMax =[]
Accuracy =[]
for L in range(0, 1):
    for P in range(0, 1):
        for l in range(0, 30000):
            print(l)
            Distance = dist(X_Test[l], X_Train)
            A = sort_train_labels_knn(Distance)
            if(Y_Test[l]) != 0 and Probability(A,Y_Train,K, Threshold) != 0:
                total = total +1
                if(Y_Test[l] == Probability(A,Y_Train,K, Threshold)):
                    counter = counter + 1
        totalM[P].append(total)
        counterM[P].append(counter)
        meanM[P].append(counter/(total+0.00001))
        counter =0
        total =0
        K= K+2
        print(K)
    K =1
    Threshold = Threshold + 1
print(totalM)
print(counterM)
print(meanM)


