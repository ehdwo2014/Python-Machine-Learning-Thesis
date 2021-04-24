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
T = 80000


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
for eachLine in dataArray:
    if len(eachLine) >1:
        x,y = eachLine.split(',')
        xar.append(float(x))
        yar.append(float(y))
print(yar)

B =0


while True:


    # connect to MetaTrader 5
    MT5Initialize()
    # wait till MetaTrader 5 connects to the trade server
    MT5WaitForTerminal()

    # set time zone to UTC
    timezone = pytz.timezone("Etc/UTC")
    # create 'datetime' object in UTC time zone to avoid the implementation of a local time zone offset
    utc_from = datetime(2019, 8, 15, tzinfo=timezone)
    # get 10 EURUSD H4 bars starting from 01.04.2019 in UTC time zone
    rates = MT5CopyRatesFrom("GBPUSD", MT5_TIMEFRAME_M5, utc_from, T)

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
    for i in range(0,T):
        vec.append(rates[i][1])

    print(vec)
    npa = np.asarray(vec, dtype=np.float32)
    f = open("dataTrain.txt", "w+")
    for i in range(T):
        f.write(str(vec[i]) + ',' + str(i) + '\n')
    f.close()

    animation.FuncAnimation(fig,animate(i), interval =1000)
    plt.pause(0.05)
    time.sleep(5)

