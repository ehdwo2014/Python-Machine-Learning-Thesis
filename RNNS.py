from keras.models import Sequential # 케라스의 Sequential()을 임포트
from keras.layers import Dense # 케라스의 Dense()를 임포트
from keras import optimizers # 케라스의 옵티마이저를 임포트
import numpy as np # Numpy를 임포트
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout
import random
from tensorflow.python.keras.initializers import RandomUniform
TrainLen= 10000

pullData = open('RSIAUD.txt', 'r').read()
dataArray = pullData.split('\n')
x1ar =[]
x2ar =[]
x3ar =[]
x4ar =[]
x5ar =[]
x6ar =[]
x7ar =[]
x8ar =[]
x9ar =[]
x10ar =[]
x11ar =[]
x12ar =[]
x13ar =[]

yar=[]
qar =[]
war=[]
ear =[]
rar=[]
zar=[]
for eachLine in dataArray:
    if len(eachLine) >1:
        x1,x2,x3,x4,x5, x6, x7, x8, x9, x10, x11, x12, x13, y, z = eachLine.split(",")
        x1ar.append(float(x1))
        x2ar.append(float(x2))
        x3ar.append(float(x3))
        x4ar.append(float(x4))
        x5ar.append(float(x5))
        x6ar.append(float(x6))
        x7ar.append(float(x7))
        x8ar.append(float(x8))
        x9ar.append(float(x9))
        x10ar.append(float(x10))
        x11ar.append(float(x11))
        x12ar.append(float(x12))
        x13ar.append(float(x13))
        yar.append(float(y))
        zar.append(float(z))


Y=[]

for i in range (len(yar)):
    if yar[i] == 0:
        yar[i] = 0.5
    if yar[i] == -1:
        yar[i] = 0

print(yar)
TrainSet = [[] for j in range(TrainLen)]
for i in range(0, TrainLen):
    TrainSet[i].append(x1ar[i]*0.01)
    TrainSet[i].append(x2ar[i]*0.01)
    TrainSet[i].append(x3ar[i]*0.01)
    TrainSet[i].append(x4ar[i]*0.01)
    TrainSet[i].append(x5ar[i]*0.01)
    TrainSet[i].append(x6ar[i] * 0.01)
    TrainSet[i].append(x7ar[i] * 0.01)
    TrainSet[i].append(x8ar[i] * 0.01)
    TrainSet[i].append(x9ar[i] * 0.01)
    TrainSet[i].append(x10ar[i] * 0.01)
    TrainSet[i].append(x11ar[i] * 0.01)
    TrainSet[i].append(x12ar[i] * 0.01)
    TrainSet[i].append(x13ar[i] * 0.01)
S = np.array(TrainSet)


model = Sequential()
model.add(SimpleRNN(128, input_shape=(FL, 1), return_sequences = True))
model.add(SimpleRNN(128, input_shape=(FL, 1)))
model.add(Dense(32, activation="sigmoid"))
model.add(Dense(1, kernel_initializer=RandomUniform(minval =-0, maxval = 0), activation="tanh"))
model.compile(loss='mse', optimizer=Adam(lr=0.00001))