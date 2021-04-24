from datetime import datetime
from pytz import timezone
import numpy as np
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import zmq
from time import sleep
from pandas import DataFrame, Timestamp
from threading import Thread
import statistics
pullData = open('BTCUSD.txt', 'r').read()
dataArray = pullData.split('\n')

xar=[]
yar=[]
zar=[]

for eachLine in dataArray:
    if len(eachLine) >1:
        x,y,z,q,w,e = eachLine.split(",")
        xar.append((x))
        yar.append(float(y))
        zar.append(float(e))



Time=[]
for i in range(0,len(zar)):
    times = datetime.datetime.strptime(xar[i],'%Y-%m-%d %H:%M:%S')
    Time.append(times)


plt.figure(1)
plt.plot(Time,zar)
plt.grid()
plt.legend()
plt.xlabel('time')
plt.ylabel('Volume')
plt.show()
