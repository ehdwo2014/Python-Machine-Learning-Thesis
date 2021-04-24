from datetime import datetime
from time import sleep
import numpy as np
import pandas as pd
import pytz
from MetaTrader5 import *
from pytz import timezone

utc_tz = timezone('UTC')
MT5Initialize()
MT5WaitForTerminal()



while True:

    timezone = pytz.timezone("Etc/UTC")
    utc_from = datetime(2020, 1, 1, tzinfo=timezone)
    rates = MT5CopyRatesFrom("AUDCAD", MT5_TIMEFRAME_H1, utc_from, 1000)
    rates_frame = pd.DataFrame(list(rates),
                               columns=['time', 'open', 'low', 'high', 'close', 'tick_volume', 'spread', 'real_volume'])
    UTC_OFFSET_TIMEDELTA = datetime.utcnow() - datetime.now()
    def local_to_utc(dt):
        return dt + UTC_OFFSET_TIMEDELTA
    rates_frame['time'] = rates_frame.apply(lambda rate: local_to_utc(rate['time']), axis=1)
    x_time = [x.time.astimezone(utc_tz) for x in rates]
    vec = []
    time =[]

    for i in range(0,1000):
        vec.append(rates[i][4])
        time.append(rates[i][0])


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
    npa = np.asarray(vec, dtype=np.float32)
    RSI2 = RSI(npa, 2)
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
    RSI14 = RSI(npa, 14)
    f = open("RawData.txt", "w+")

    f.write(str(RSI2[len(vec)-1]) +',' + str(RSI3[len(vec)-1]) +', '+ str(RSI4[len(vec)-1]) +',' + str(RSI5[len(vec)-1])+','  +str(RSI6[len(vec)-1]) +',' + str(RSI7[len(vec)-1])+','  + str(RSI8[len(vec)-1]) +',' + str(RSI9[len(vec)-1])+','  + str(RSI10[len(vec)-1]) +',' + str(RSI11[len(vec)-1])+','  + str(RSI12[len(vec)-1]) +',' + str(RSI13[len(vec)-1])+','   + str(RSI14[len(vec)-1]) +'\n')
    f.close()
    print(vec[len(vec)-1])
    print(RSI14[len(RSI14)-1])
    sleep(1)  # Time in seconds
    vec.clear()
    time.clear()
