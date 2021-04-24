import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
pullData = open('AUDCADForRNN.txt', 'r').read()
dataArray = pullData.split('\n')
xar = []
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout

TrainLen = 10000

dif =[]
PercentageDif=[]
for eachLine in dataArray:
    if len(eachLine) > 1:
        x, y = eachLine.split(",")
        xar.append(float(x))


for i in range(0, len(xar)-1):
    PercentageDif.append((xar[i] - xar[i+1])/ xar[i+1])

diff =[]

for l in range(0,len(xar)-1):
        if (xar[i] - xar[i+1]) < 0:
            diff.append(1)
        if xar[i] - xar[i+1] > 0:
            diff.append(-1)
        if xar[i] - xar[i+1]  == 0:
            diff.append(0)

for i in range(0, len(PercentageDif)):
    if PercentageDif[i] > 0.01:
        PercentageDif[i] = 0.01
    if PercentageDif[i] < -0.01:
        PercentageDif[i] = -0.01

series = Series(PercentageDif)
print(series)
values = series.values
values = values.reshape((len(values), 1))
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(values)
normalized = scaler.transform(values)
inversed = scaler.inverse_transform(normalized)



s = np.array(normalized[0:TrainLen])
t = np.array(normalized[10000:20000])

FL = 6

S = np.fliplr(toeplitz(np.r_[s[-1], np.zeros(s.shape[0] - FL)], s[::-1]))

X_train = S[:-1, :FL][:, :, np.newaxis]
Y_train = S[:-1, FL]
T = np.fliplr(toeplitz(np.r_[t[-1], np.zeros(t.shape[0] - FL)], t[::-1]))

X_val = T[:-1, :FL][:, :, np.newaxis]
Y_val = T[:-1, FL]
print("here")

YB = Y_val.reshape(1, -1)
InY = scaler.inverse_transform(YB)

print(X_train.shape)
print(X_train[0:2][0][0])
print(Y_train.shape)

from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.layers import LSTM
from keras import optimizers
from tensorflow.python.keras.initializers import RandomUniform
from keras.optimizers import SGD
from keras.optimizers import Adam

from sklearn import preprocessing
from keras.layers import LSTM, Dense, Bidirectional, Input,Dropout,BatchNormalization, CuDNNGRU, CuDNNLSTM


model = Sequential()
model.add(SimpleRNN(128, input_shape=(FL, 1), return_sequences = True))
model.add(SimpleRNN(128, input_shape=(FL, 1)))
model.add(Dense(32, activation="sigmoid"))
model.add(Dense(1, kernel_initializer=RandomUniform(minval =-0, maxval = 0), activation="tanh"))
model.compile(loss='mse', optimizer=Adam(lr=0.002))


history = model.fit(X_train, Y_train, nb_epoch=500, batch_size=500, verbose = 1)

A = (model.predict(X_val[:, :, :]).reshape(1,-1))
InX = scaler.inverse_transform(A)

T =0
R = 0
U=0
D=0
UR=0
UD=0
print(InY[0])
print(InX[0])

Q = []

for i in range(len(Y_val)):
    if InX[0][i] > 0:
        T = T + 1
        U = U+1
        if InY[0][i] > 0:
            R=R+1
            UR = UR +1
    if InX[0][i] < -0:
        T = T + 1
        D = D+1
        if InY[0][i] < 0:
            R = R + 1
            UD = UD +1


print(R)
print(T)
print(U)
print(D)
print(UR)
print(UD)

Final = []
S =0
for i in range(0, len(Q)):
    S = Q[i] + S
    Final.append(S)

model_json = model.to_json()
with open("model.json", "w") as json_file :
    json_file.write(model_json)



model.save_weights("model.h5")
print("Saved model to disk")

f = open("Proper.txt", "w+")

for i in range(len(Final)):
    f.write(str(Final[i])+ '\n')