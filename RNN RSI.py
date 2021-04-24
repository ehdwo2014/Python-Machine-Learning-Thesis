from keras.models import Sequential # 케라스의 Sequential()을 임포트
from keras.layers import Dense # 케라스의 Dense()를 임포트
from keras import optimizers # 케라스의 옵티마이저를 임포트
import numpy as np # Numpy를 임포트
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout
import random
from tensorflow.python.keras.initializers import RandomUniform
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.layers import LSTM
from keras import optimizers
from tensorflow.python.keras.initializers import RandomUniform
from keras.optimizers import SGD
from keras.optimizers import Adam
from sklearn import preprocessing
from keras.layers import LSTM, Dense, Bidirectional, Input,Dropout

TrainLen= 10000

pullData = open('RRRRSI.txt', 'r').read()
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



S = np.expand_dims(TrainSet, axis=2)
print(np.shape(S))

model=Sequential()
model.add(SimpleRNN(16,  input_shape=(13, 3), return_sequences= True,  activation='tanh'))
model.add(SimpleRNN(8, activation ='tanh'))
model.add(Dense(1, activation="sigmoid"))
sgd=optimizers.SGD(lr=0.01)
model.compile(optimizer=sgd ,loss='binary_crossentropy',metrics=['binary_accuracy'])
# 옵티마이저는 경사하강법의 일종인 확률적 경사 하강법 sgd를 사용합니다.
# 손실 함수(Loss function)는 binary_crossentropy(이진 크로스 엔트로피)를 사용합니다.
model.fit(S ,yar[0:TrainLen], batch_size=640, epochs=200, shuffle=False)
# 주어진 X와 y데이터에 대해서 오차를 최소화하는 작업을 200번 시도합니다.





Test_Length = 10000
Test = [[] for j in range(Test_Length)]
Y_Test =[]
P_Test =[]

for i in range(0, Test_Length):
    Test[i].append(x1ar[TrainLen+i]*0.01)
    Test[i].append(x2ar[TrainLen+i]*0.01)
    Test[i].append(x3ar[TrainLen+i]*0.01)
    Test[i].append(x4ar[TrainLen+i]*0.01)
    Test[i].append(x5ar[TrainLen+i]*0.01)
    Test[i].append(x6ar[TrainLen+i]*0.01)
    Test[i].append(x7ar[TrainLen+i]*0.01)
    Test[i].append(x8ar[TrainLen+i]*0.01)
    Test[i].append(x9ar[TrainLen+i]*0.01)
    Test[i].append(x10ar[TrainLen+i]*0.01)
    Test[i].append(x11ar[TrainLen+i]*0.01)
    Test[i].append(x12ar[TrainLen+i]*0.01)
    Test[i].append(x13ar[TrainLen+i]*0.01)

for i in range(0, Test_Length):
    Y_Test.append(yar[i+TrainLen])
    P_Test.append(zar[i+TrainLen])


Q = np.expand_dims(Test, axis=2)
MP = model.predict(Q)
T=0
R=0

Profit=[]
RU = 0
RD =0
U =0
D = 0

Threshold = 0.47
for i in range(0, Test_Length):
    print(i)
    if MP[i] > 1-Threshold and Y_Test[i] != 0.5:
        T = T+1
        U = U + 1
        if Y_Test[i] == 1:
            R = R+1
            RU = RU + 1
    if MP[i]  < Threshold and Y_Test[i] != 0.5:
        T = T + 1
        D = D+1
        if Y_Test[i] == 0:
            R = R + 1
            RD = RD + 1

ProfitF =0
Index =[]
for i in range(0, len(MP)):
    if MP[i] > 1-Threshold and Y_Test[i] != 0.5:
        if Y_Test[i] == 1:
            ProfitF = abs(P_Test[i]) + ProfitF
            Profit.append(ProfitF*100000)
            Index.append(i)
        if Y_Test[i] == 0:
            ProfitF = -abs(P_Test[i]) + ProfitF
            Profit.append(ProfitF * 100000)
            Index.append(i)
    if MP[i] < Threshold and Y_Test[i] != 0.5:
        if Y_Test[i] == 0:
            ProfitF = abs(P_Test[i]) + ProfitF
            Profit.append(ProfitF * 100000)
            Index.append(i)
        if Y_Test[i] == 1:
            ProfitF = -abs(P_Test[i]) + ProfitF
            Profit.append(ProfitF * 100000)
            Index.append(i)
print(Profit)

f = open("Compare2.txt", "w+")
for i in range(len(Profit)):
    f.write(str(Profit[i]) + ',' + str(Index[i]) + '\n')
f.close()


print(T)
print(R)
print(U)
print(RU)
print(D)
print(RD)