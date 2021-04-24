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


k = 1
L = 200
PS = 200
TS=30000
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
Index = 4000

utc_tz = timezone('UTC')

pd.set_option('display.max_columns', 500)  # number of columns to be displayed
pd.set_option('display.width', 1500)  # max table width to display
# import pytz module for working with time zone
import pytz
init = 0

xar= []
yar =[]
zar =[]

pullData = open('dataTrain.txt', 'r').read()
dataArray = pullData.split('\n')
for eachLine in dataArray:
    if len(eachLine) >1:
        x,y,z = eachLine.split(",")
        xar.append(float(x))
        yar.append(float(y))
        zar.append(float(z))

TrainSet = [[] for j in range(TS)]
ValidationSet =[[] for p in range(TS)]
TrainSetA = [[] for j in range(TS)]
ValidationSetA =[[] for p in range(TS)]
for l in range(0,TS):
    for p in range(0,L):
        TrainSet[l].append(xar[l+p])
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

for p in range(0,L):
    DataForPredict.append(xar[TS+Index+1+p])
    DataForPredictA.append(xar[TS+Index+p])

DataForVal.append(0)
for t in range(0,PS):
    DataForVal.append(xar[TS+L+Index+1+t] - xar[TS+L+Index])
    DataForValA.append(xar[TS+L+t+Index])
    if(t == PS-1):
        DataForValA.append(xar[TS+L+t+Index+1])




def compare(x1,x2):
    V =[]
    for p in range(0,L):
        V.append(abs(x2[p] - x1[p]))
    return sum(V)

Min =[]


for t in range(0,TS):
    Min.append((compare(TrainSet[t],DataForPredict),t))

Min.sort(reverse = False)

VecAV = [[] for j in range(PS)]
print(VecAV)

for l in range(0,PS):
    for s in range(0,k):
        VecAV[l].append(ValidationSet[Min[s][1]][l])


AVG = []
AVG.append(0)

for l in range(0,PS):
    AVG.append(statistics.mean(VecAV[l]))
print(AVG)
Prediction= []
print(DataForValA)
print(AVG)
A =DataForValA[0]

for l in range(0,PS+1):
    Prediction.append(A + AVG[l])
print(Prediction)


plt.figure(1)
plt.plot(range(0,L),TrainSet[Min[0][1]])
plt.plot(range(0,L),DataForPredict, 'r--')



plt.show()
plt.figure(2)
plt.plot(range(0,PS),ValidationSet[Min[0][1]])



plt.plot(range(0,PS+1), AVG)
plt.plot(range(0,PS+1),DataForVal, 'r--')
plt.show()

plt.figure(3)
plt.plot(range(0,PS+1), Prediction, 'b--')
plt.plot(range(0,PS+1), DataForValA, 'r--')
plt.show()