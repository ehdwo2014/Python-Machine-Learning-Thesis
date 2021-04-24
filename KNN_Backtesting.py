import numpy as np
import scipy
from scipy.spatial import distance

Final_T = []
Final_R = []
Final_M = []


def dist(X, TrainX):
    Dist = np.zeros(len(TrainX))
    for i in range(0, len(TrainX)):
        Dist[i] = np.sqrt(np.sum((X - TrainX[i]) ** 2))
    return Dist


def sort_train_labels_knn(Dist):
    MatrixIndex = []
    for i in range(0, TrainLen):
        MatrixIndex.append((Dist[i], int(i)))
    MatrixIndex.sort()
    a = np.array(MatrixIndex)
    return a


def Probability(sorted_Index, Y, K, Th):
    UD = []
    for i in range(0, K):
        UD.append(Y[int(sorted_Index[i][1])])
    if sum(UD) / K > Th / 10:
        return 1
    if sum(UD) / K < -Th / 10:
        return -1
    else:
        return 0

T = 19000
feature_length = 3
TrainLen = 1000
TestLen = 5000
ValLen = 0
Threshold = 1
Time_lag = 1
K = 1


pullData = open('KNNAUD.txt', 'r').read()
dataArray = pullData.split('\n')
xar = []




for eachLine in dataArray:
    if len(eachLine) > 1:
        x, y = eachLine.split(",")
        xar.append(float(x))

for N in range (0,9):
    for Q in range(0,7):
        for W in range(0, 1):
            X_Train = np.zeros((TrainLen, feature_length))
            Y_Train = np.zeros((TrainLen, 1))
            X_Test = np.zeros((TestLen, feature_length))
            Y_Test = np.zeros((TestLen, 1))
            Profit = np.zeros((TestLen, 1))
            X_Val = np.zeros((ValLen, feature_length))
            Y_Val = np.zeros((ValLen, 1))
            ProfitA =[]
            diff =[]
            for l in range(0,T):
                if xar[l+feature_length] - xar[l+feature_length+Time_lag] < 0:
                    print(l)
                    diff.append(1)
                if xar[l+feature_length] - xar[l+feature_length+Time_lag] > 0:
                    diff.append(-1)
                if xar[feature_length+l] - xar[l+feature_length+Time_lag]  == 0:
                    diff.append(0)
            for l in range(0,T):
                 ProfitA.append(xar[l+feature_length+Time_lag]-xar[l+feature_length])
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

            for i in range(TrainLen, TrainLen+TestLen):
                for l in range(0,1):
                     Profit[TrainLen - i][l] =  ProfitA[i+l]

            Distance = np.zeros((TrainLen))
            Index = np.zeros((TrainLen))




            counter = 0
            total =0
            mean =0
            totalM =[[] for i in range (0, 1)]
            counterM =[[] for i in range (0, 1)]
            meanM =[[] for i in range (0, 1)]
            print(counterM)
            counterMax =[]
            Accuracy =[]
            AmountP = []

            for L in range(0, 1):
                for P in range(0, 1):
                    for l in range(0, 5000):
                        print(l)
                        Distance = dist(X_Test[l], X_Train)
                        A = sort_train_labels_knn(Distance)
                        if Probability(A, Y_Train, K, Threshold) != 0:
                            total = total + 1
                            if (Y_Test[l] == Probability(A, Y_Train, K, Threshold)):
                                counter = counter + 1
                                AmountP.append(((abs(round(float(Profit[l]), 6) * 100000)),l))
                            if (Y_Test[l] != Probability(A, Y_Train, K, Threshold)):
                                AmountP.append(((-abs(round(float(Profit[l]), 6) * 100000)), l))
                    totalM[P].append(total)
                    counterM[P].append(counter)
                    meanM[P].append(counter/(total+0.00001))
                    counter =0
                    total =0
            print(Threshold)
            print(K)
            print(totalM)
            print(counterM)
            print(meanM)
            Final_T.append(totalM)
            Final_R.append(counterM)
            Final_M.append(meanM)


            Cumulative =[]
            Cumu = 0
            Sum = 0
            for i in range(0, len(AmountP)):
                Cumu = Cumu + AmountP[i][0]
                Cumulative.append((Cumu, AmountP[i][1]))

            print(W)
            print(P)
            f = open('BackTestKNN'+str(W) + str(Q) +str(N) +'.txt', "w+")
            for i in range(0, len(AmountP)):
                f.write(str(Cumulative[i][0]) + ',' + str(Cumulative[i][1]) + '\n')
            f.close()
            ProfitA.clear()
            diff.clear()

            feature_length = feature_length + 1
        feature_length = 3
        Threshold = Threshold + 1
    Threshold = 1
    K = K+2

print(Final_T)
print(Final_R)
print(Final_M)
