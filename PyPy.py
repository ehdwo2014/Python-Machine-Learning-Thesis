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


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

T = 75000

def RSI(prices, n =2):
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


def Difference(Close, Open):
    Diff =[]
    for i in range(len(Close)):
        Diff.append(round(Close[i] - Open[i],6))
    return Diff



utc_tz = timezone('UTC')

pd.set_option('display.max_columns', 500)  # number of columns to be displayed
pd.set_option('display.width', 1500)  # max table width to display
# import pytz module for working with time zone
import pytz
init = 0

pullData = open('dataTrain.txt', 'r').read()
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
utc_from = datetime(2019, 9, 11, tzinfo=timezone)
# get 10 EURUSD H4 bars starting from 01.04.2019 in UTC time zone
rates = MT5CopyRatesFrom("AUDCAD", MT5_TIMEFRAME_H1, utc_from, T+40)

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

for i in range(0,T+40):
    vec.append(rates[i][4])
    time.append(rates[i][0])
    opens.append(rates[i][1])
    high.append(rates[i][2])
    low.append(rates[i][3])
    close.append(rates[i][4])
    adc.append(rates[i][4])
    volume.append(rates[i][5])

Dif = Difference(close, opens)

DD =[]



DifDif =[]

for l in range(0,T+20):
    print(l)
    if vec[l] - vec[l+1] < 0:
        diff.append(1)
    if vec[l] - vec[l+1] > 0:
        diff.append(-1)
    if vec[l] - vec[l+1]  == 0:
        diff.append(0)
    DifDif.append(Dif[l+8])

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

Pin = RSI(npa, 14)
Pia = RSI(npa, 2)

print(len(vec))
print(Pin)


AdSum =[]
MiSum =[]
AdP =[]
RS = []
print(seq)




npa = np.asarray(vec, dtype=np.float32)
f = open("RSI3.txt", "w+")

for i in range(T):
    f.write(str(round(Pin[i], 0)) +',' + str(round(Pia[i], 0)) + ',' + str(diff[i]) + '\n')
f.close()

print(len(Pin))
print(len(diff))
print(vec)
print(vec[0])
print(Pin)
print(Pin[0])
print(Dif)
print(seq)
print(diff)
