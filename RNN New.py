import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
pullData = open('AUDCAD1H.txt', 'r').read()
dataArray = pullData.split('\n')
xar = []
from pandas import Series
from sklearn.preprocessing import MinMaxScaler

feature_length = 5
FL = feature_length
TrainLen = 300
TestLen = 5000

X_train = np.zeros((TrainLen, feature_length))
Y_train = np.zeros((TrainLen, 1))
X_test = np.zeros((TestLen, feature_length))
Y_test = np.zeros((TestLen, 1))


dif =[]
PercentageDif=[]
for eachLine in dataArray:
    if len(eachLine) > 1:
        x, y, z, e, p, w = eachLine.split(",")
        xar.append(float(z))

diff =[]

for i in range(0, len(xar)-1):
    PercentageDif.append((xar[i] - xar[i+1])/ xar[i+1])
for i in range(0, len(PercentageDif)-1):
    if PercentageDif[i+1] > 0:
        diff.append(1)
    if PercentageDif[i+1] < 0:
        diff.append(0)
    if PercentageDif[i+1] == 0:
        diff.append(0.5)

for i in range(0, TrainLen):
    for l in range(0, feature_length):
        X_train[i][l] = (xar[i + l] - xar[i + l + 1]) / xar[i + l + 1]
for i in range(0, TrainLen):
    for l in range(0,1):
        Y_train[i][l] = diff[i + feature_length -1]

for i in range(TrainLen, TrainLen+TestLen):
    for l in range(0, feature_length):
        X_test[TrainLen - i][l] = (xar[i+l] - xar[i+l+1]) / xar[i+l+1]
for i in range(TrainLen, TrainLen+TestLen):
    for l in range(0,1):
        Y_test[TrainLen - i][l] =  diff[i+feature_length - 1]


x = np.expand_dims(X_train, axis=2)
print(x.shape)
x_v = np.expand_dims(X_test, axis=2)

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
model.add(Dense(32, activation="sigmoid"))
model.add(Dense(1, kernel_initializer=RandomUniform(minval =-0, maxval = 0), activation="tanh"))
model.compile(loss='mse', optimizer=Adam(lr=0.00001))

history = model.fit(x, Y_train, nb_epoch=200, batch_size=200, verbose = 1)


plt.plot(history.history["loss"])
plt.title("Loss")
print(model.predict(x)[0:10])
print(Y_train[0:10])

Z = []
C = Y_test[0:TestLen]
M = model.predict(x_v)
T =0

B =0
S =0

for i in range(0,TestLen):
    if M[i] > 0.5:
        Z.append(1)
        B=B+1
    if M[i] < 0.5:
        Z.append(0)
        S=S+1
for i in range(0,TestLen):
    if Z[i] == C[i]:
        T = T+1

print(T)
print(B)
print(S)
