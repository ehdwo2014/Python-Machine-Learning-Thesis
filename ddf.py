from datetime import datetime
from MetaTrader5 import *
from pytz import timezone
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import zmq
from time import sleep
from pandas import DataFrame, Timestamp
from threading import Thread
import statistics


L = 7
TS=2000
T =3000
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
Index = 0
VL = 1
k=1
HM = 5


def compare(x1,x2):
    V =[]
    for p in range(0,L):
        V.append(abs(x2[p] - x1[p]))
    return sum(V)

pullData = open('dataTrain.txt', 'r').read()
dataArray = pullData.split('\n')
xar= []
yar =[]
zar =[]
for eachLine in dataArray:
    if len(eachLine) >1:
        x,y,z = eachLine.split(",")
        xar.append(float(x))
        yar.append(float(y))
        zar.append(float(z))
TrainSet = []
ValidationSet =[]
TrainSetA = []
ValidationSetA =[]
Feature =[[] for i in range(0,T)]
counter = 0
Min=[[] for j in range(0,T-TS)]
counterA =[]
Total=[]
FinalR = []
for q in range(0,HM):
    print(q)
    Feature = [[] for i in range(0, T+Index)]
    Min = [[] for j in range(0, T - TS)]

    for D in range(0, T+Index):
        for i in range(0,L):
            Feature[D].append(xar[i+D])

    for l in range(0,TS):
        TrainSet.append(Feature[l])
        ValidationSet.append(yar[l])
    for t in range(TS+Index,T+Index):
        TrainSetA.append(Feature[t])
        ValidationSetA.append(yar[t])



    for l in range(0,T-TS):
        for t in range(0,TS):
            Min[l].append((compare(TrainSet[t], TrainSetA[l]) , t))


    for w in range(0, len(Min)):
        Min[w].sort(reverse = False)


    Sum=[]

    TC =0
    for p in range(len(Min)):
        for a in range(0, k):
            Sum.append(ValidationSet[Min[p][a][1]])
        if abs(sum(Sum))/k > 0:
            TC = TC + 1
            if ValidationSetA[p] == 1:
                if sum(Sum) > 0:
                    counter = counter +1
                    print(1)
            if ValidationSetA[p] == -1:
                if sum(Sum) < 0:
                    print(-1)
                    counter = counter + 1
        Sum.clear()



    counterA.append(counter)
    Total.append(TC)
    TC =0
    Feature.clear()
    counter = 0
    Min.clear()
    TrainSet.clear()
    ValidationSet.clear()
    TrainSetA.clear()
    ValidationSetA.clear()
    Index = Index+1000

k = k + 1
Z =0
T = 0
Index =0
for l in range(0,HM):
    Z = counterA[l]+ Z
    T = Total[l] + T

print(counterA)
print(Total)


for i in range(0,len(counterA)):
    FinalR.append(counterA[i] / Total[i])



print(FinalR)
print(sum(FinalR)/len(FinalR))