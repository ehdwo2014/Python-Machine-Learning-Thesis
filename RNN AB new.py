import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
pullData = open('AUDCADForRNN.txt', 'r').read()
dataArray = pullData.split('\n')

from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout

feature_length = 3
FL = feature_length
Time_lag = 1
TrainLen = 1000
TestLen = 10000

X_train = np.zeros((TrainLen, feature_length))
Y_train = np.zeros((TrainLen, 1))
X_test = np.zeros((TestLen, feature_length))
Y_test = np.zeros((TestLen, 1))

xar = []
yar =[]
dif =[]
PercentageDif=[]
for eachLine in dataArray:
    if len(eachLine) > 1:
        x, y,= eachLine.split(",")
        xar.append(float(x))
        yar.append(float(y))
diff=[]
PercentageDif=[]

for i in range(0, len(xar)-1):
    PercentageDif.append((xar[i+1] - xar[i])/ xar[i])




for i in range(0, len(PercentageDif)-1):
    if PercentageDif[i+1] > 0:
        diff.append(1)
    if PercentageDif[i+1] < 0:
        diff.append(0)
    if PercentageDif[i+1] == 0:
        diff.append(0.5)

print(PercentageDif[0:10])
print(xar[0:10])
series = Series(PercentageDif)
values = series.values
values = values.reshape((len(values), 1))
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(values)
Nxar = scaler.transform(values)
inversed = scaler.inverse_transform(Nxar)


for i in range(0, TrainLen):
    for l in range(0, feature_length):
        X_train[i][l] = Nxar[i+l]
for i in range(0, TrainLen):
    Y_train[i] = Nxar[i+feature_length+Time_lag]

for i in range(0, TestLen):
    for l in range(0, feature_length):
        X_test[i][l] = Nxar[i+l+TrainLen]
for i in range(0, TestLen):
    Y_test[i] = Nxar[i+feature_length+Time_lag+TrainLen]
X = np.expand_dims(X_train, axis=2)
Xv = np.expand_dims(X_test, axis=2)
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.layers import LSTM
from keras import optimizers
from tensorflow.python.keras.initializers import RandomUniform
from keras.optimizers import SGD
from keras.optimizers import Adam
from sklearn import preprocessing
from keras.layers import LSTM, Dense, Bidirectional, Input,Dropout
model = Sequential()
model.add(SimpleRNN(128, input_shape=(FL, 1), return_sequences = True))
model.add(SimpleRNN(128, input_shape=(FL, 1)))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, kernel_initializer=RandomUniform(minval =-0, maxval = 0), activation="sigmoid"))
model.compile(loss='mean_squared_logarithmic_error', optimizer=Adam(lr=0.00003))

history = model.fit(X, Y_train, nb_epoch=1000, batch_size=1024, verbose = 1)
M = (model.predict(Xv))
Mi = scaler.inverse_transform(M)
Vy = scaler.inverse_transform(Y_test)
print(Mi)
print(Vy)
T=0
R=0
U =0
D =0
for i in range(0, len(Vy)):
    if Mi[i] > 0.0001 and Mi[i] != 0 and Vy[i] != 0:
        T = T+1
        U = U+1
        if Vy[i] > 0 :
            R = R+1
    if Mi[i] < -0.0001 and Mi[i] != 0 and Vy[i] != 0:
        T = T + 1
        D = D+1
        if Vy[i] < 0:
            R = R + 1
print(T)
print(R)
print(U)
print(D)