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
import matplotlib.pyplot as plt

#getting data
pullData = open('Logistic.txt', 'r').read()
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
#Set the input vectors

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
# Import feature and output

Y=[]

for i in range (len(yar)):
    if yar[i] == 0:
        yar[i] = 0.5
    if yar[i] == -1:
        yar[i] = 0

print(yar)
TrainSet = [[] for j in range(TrainLen)]

# Normalization. Since RSI is value is 0 to 100, the normalized value will be 0 to 1
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

model=Sequential()
model.add(Dense(1,  kernel_initializer=RandomUniform(minval =1, maxval = 1), input_dim=13, activation='sigmoid'),)
sgd=optimizers.SGD(lr=0.01)
model.compile(optimizer=sgd ,loss='binary_crossentropy',metrics=['binary_accuracy'])
# stochastic gradient decent algorithm will be used for optimization process
# Loss function will be binary_crossentropy
history = model.fit(S,yar[0:TrainLen], batch_size=640, epochs=2000, shuffle=False)
# 2000 epochs

plt.plot(history.history["binary_accuracy"])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

Test_Length = 10000
Test = [[] for j in range(Test_Length)]
Y_Test =[]
P_Test =[]

#set the test data
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

# variable and vectors for accuracy

Q = np.array(Test)
MP = model.predict(Q)
T=0
R=0
U=0
UR =0
D =0
DR =0

Profit=[]

# threshold. 0.46 means it checks only when the signal is between 0.54 ~ 0.46
Threshold = 0.46
for i in range(0, len(MP)):
    if MP[i] > 1-Threshold and Y_Test[i] != 0.5:
        T = T+1
        U = U+1
        if Y_Test[i] == 1:
            R = R+1
            UR = UR+1
    if MP[i]  < Threshold and Y_Test[i] != 0.5:
        T = T + 1
        D = D+1
        if Y_Test[i] == 0:
            R = R + 1
            DR = DR + 1
ProfitF =0
Index =[]
for i in range(0, len(MP)):
    if MP[i] > 1-Threshold and Y_Test[i] != 0.5:
        if Y_Test[i] == 1:
            ProfitF = abs(P_Test[i]) + ProfitF
            Profit.append(ProfitF*100000) # * 100000 for making the profit in pip unit
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
from keras.models import load_model
# Save in textfile for backtesting
f = open("BackTestLR0000.txt", "w+")
for i in range(len(Profit)):
    f.write(str(Profit[i]) + ',' + str(Index[i]) + '\n')
f.close()
# Save the model for real time trade signal
model.save('my_model.h5')

print(T)
print(R)
print(U)
print(UR)
print(D)
print(DR)