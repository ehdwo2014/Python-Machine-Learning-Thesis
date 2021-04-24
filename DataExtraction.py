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

T = 32000 # number of data to extract

def RSI(prices, n =2): # Relative strength index function
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1]
        if delta >0:
            upval = delta
            downval = 0.

        else:
            upval =0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n
        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)
    return rsi


utc_tz = timezone('UTC') # Set time zone

pd.set_option('display.max_columns', 500)  # number of columns to be displayed
pd.set_option('display.width', 1500)  # max table width to display
# import pytz module for working with time zone
import pytz
init = 0

pullData = open('dataTrain.txt', 'r').read() #
dataArray = pullData.split('\n')
xar=[]
yar=[]


# connect to MetaTrader 5
MT5Initialize()
# wait till MetaTrader 5 connects to the trade server
MT5WaitForTerminal()

# set time zone to UTC
timezone = pytz.timezone("Etc/UTC")
# create 'datetime' object in UTC time zone to avoid the implementation of a local time zone offset
utc_from = datetime(2019, 8, 1, tzinfo=timezone)
# get 10 EURUSD H4 bars starting from 01.04.2019 in UTC time zone
rates = MT5CopyRatesFrom("AUDCAD", MT5_TIMEFRAME_H1, utc_from, T+100)

# shut down connection to MetaTrader 5
MT5Shutdown()
# display each element of obtained data in a new line
print("Display obtained data 'as is'")
for rate in rates:
    print(rate)

# create DataFrame out of the obtained data
rates_frame = pd.DataFrame(list(rates),
                           columns=['time', 'open', 'low', 'high', 'close', 'tick_volume', 'spread', 'real_volume'])

# display data
print("\nDisplay dataframe with data")
print(rates_frame)  # we can see that Python provides bars open time in the local time zone with an offset

# get a UTC time offset for the local PC
UTC_OFFSET_TIMEDELTA = datetime.utcnow() - datetime.now()


# create a simple function correcting the offset head-on
def local_to_utc(dt):
    return dt + UTC_OFFSET_TIMEDELTA


# apply the offset for the 'time' column in the rates_frame dataframe
rates_frame['time'] = rates_frame.apply(lambda rate: local_to_utc(rate['time']), axis=1)

# display the data once again and make sure the H4 candles open time is now multiple of 4
print("\nDisplay the dataframe after adjusting the time")
print(rates_frame)

# vector for data
x_time = [x.time.astimezone(utc_tz) for x in rates]
vec = []
diff = []
time =[]
opens =[]
high =[]
low =[]
close =[]
adc =[]
volume =[]

for i in range(0,T+100):
    vec.append(rates[i][4])
    time.append(rates[i][0])
    opens.append(rates[i][1])
    high.append(rates[i][2])
    low.append(rates[i][3])
    close.append(rates[i][4])
    adc.append(rates[i][4])
    volume.append(rates[i][5])

DD =[]



DifDif =[]


# TimeFrameLag and input dimention
TimeLag = 11
ID = 3
InputDim = ID - 1
for l in range(0,T+30): #Making output vector
    print(l)
    if vec[l+InputDim] - vec[l+TimeLag+InputDim] < 0:
        diff.append(1)
        DD.append(vec[l+TimeLag+InputDim] - vec[l+InputDim])
    if vec[l+InputDim] - vec[l+TimeLag+InputDim]> 0:
        diff.append(-1)
        DD.append(vec[l+TimeLag+InputDim] - vec[l+InputDim])
    if vec[l+InputDim] - vec[l+TimeLag+InputDim]  == 0:
        diff.append(0)
        DD.append(vec[l+TimeLag+InputDim] - vec[l+InputDim])

seq =[]
seqC = 0
for w in range(0,T+5):
    if diff[w] == diff[w+1]:
        seq.append(seqC)
        seqC = seqC + 1
    if diff[w] != diff[w+1]:
        seqC = 0

Pin = []


npa = np.asarray(vec, dtype=np.float32)

RSI2 = RSI(npa,2) # RSI input vector for LR and RNN
RSI3 = RSI(npa, 3)
RSI4 = RSI(npa, 4)
RSI5 = RSI(npa, 5)
RSI6 = RSI(npa, 6)
RSI7 = RSI(npa, 7)
RSI8 = RSI(npa, 8)
RSI9 = RSI(npa, 9)
RSI10 = RSI(npa, 10)
RSI11 = RSI(npa, 11)
RSI12 = RSI(npa, 12)
RSI13 = RSI(npa, 13)
RSI14 = RSI(npa,14)

print(len(vec))
print(Pin)





 # Save the file in RSIRNN.txt
npa = np.asarray(vec, dtype=np.float32)
f = open("RNN.txt", "w+")

for i in range(T):
    f.write(str(RSI2[i]) +',' + str(RSI3[i]) +', '+ str(RSI4[i]) +',' + str(RSI5[i])+','  +str(RSI6[i]) +',' + str(RSI7[i])+','  + str(RSI8[i]) +',' + str(RSI9[i])+','  + str(RSI10[i]) +',' + str(RSI11[i])+','  + str(RSI12[i]) +',' + str(RSI13[i])+','   + str(RSI14[i]) +',' + str(diff[i])+','+ str(DD[i]) + '\n')
f.close()




