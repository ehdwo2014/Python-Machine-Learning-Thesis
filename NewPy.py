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


L = 5
PS = 5
TS=3000
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
Index = 500
VL = 1
k=6


def compare(x1,x2):
    V =[]
    for p in range(0,L):
        V.append(abs(x2[p] - x1[p]))
    return sum(V)

utc_tz = timezone('UTC')

pd.set_option('display.max_columns', 500)  # number of columns to be displayed
pd.set_option('display.width', 1500)  # max table width to display
# import pytz module for working with time zone
import pytz
init = 0

xar= []
yar =[]

pullData = open('dataTrain.txt', 'r').read()
dataArray = pullData.split('\n')
for eachLine in dataArray:
    if len(eachLine) >1:
        x,y,z,h = eachLine.split(",")
        xar.append(float(x))
        yar.append(float(y))


TrainSet = [[] for j in range(TS)]
ValidationSet =[[] for p in range(TS)]
TrainSetA = [[] for j in range(TS)]
ValidationSetA =[[] for p in range(TS)]
for l in range(0,TS):
    for p in range(0,L):
        TrainSet[l].append(xar[l+p+1] - xar[l])
        TrainSetA[l].append(xar[p])

    for t in range(0,PS):
        ValidationSet[l].append(xar[L+l+t+1]-xar[L+l])
        ValidationSetA[l].append(xar[L+t])


print(xar[L+1])
print(xar[L])
print(ValidationSet[0])
print(TrainSet[1])

DataForPredict = []
DataForVal =[]
DataForPredictA = []
DataForValA =[]


Min =[]
VecAV = [[] for j in range(PS)]
AVG = []
Prediction = []
counter =0
ForV = []
ForT =[]
PerC =[]

for w in range(0,VL):
    for p in range(0,L):
        DataForPredict.append(xar[TS+Index+1+p+w] - xar[TS+Index+w])
        DataForPredictA.append(xar[TS+Index+p+w])
    DataForVal.append(0)
    for t in range(0,PS):
        DataForVal.append(xar[TS+L+Index+1+t+w] - xar[TS+L+Index+w])
        DataForValA.append(xar[TS+L+t+Index+w])
        if(t == PS-1):
            DataForValA.append(xar[TS+L+t+Index+1+w])
    for t in range(0,TS):
        Min.append((compare(TrainSet[t],DataForPredict),t))

    Min.sort(reverse = False)

    for l in range(0,PS):
        for s in range(0,k):
            VecAV[l].append(ValidationSet[Min[s][1]][l])
    AVG.append(0)

    for l in range(0,PS):
        AVG.append(statistics.mean(VecAV[l]))

    A =DataForValA[0]

    for l in range(0,PS+1):
        Prediction.append(A + AVG[l])


    if Prediction[0] > Prediction[PS]:
        ForT.append(-1)
    if Prediction[0] < Prediction[PS]:
        ForT.append(1)
    if Prediction[0] == Prediction[PS]:
        ForT.append(0)
    if DataForValA[0] > DataForValA[PS]:
        ForV.append(-1)
    if DataForValA[0] < DataForValA[PS]:
        ForV.append(1)
    if DataForValA[0] == DataForValA[PS]:
        ForV.append(0)
    DataForValA.clear()
    DataForPredict.clear()
    DataForVal.clear()
    DataForPredictA.clear()
    Min.clear()
    VecAV.clear()
    AVG.clear()
    Prediction.clear()
    VecAV = [[] for j in range(PS)]
    print(w)

print(ForT)
print(ForV)
for i in range(VL):
    if ForT[i] == ForV[i]:
        counter = counter + 1
ForT.clear()
ForV.clear()
PerC.append(counter)
counter = 0

print(PerC)

