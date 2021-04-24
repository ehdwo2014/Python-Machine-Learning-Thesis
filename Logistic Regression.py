from keras.models import Sequential # 케라스의 Sequential()을 임포트
from keras.layers import Dense # 케라스의 Dense()를 임포트
from keras import optimizers # 케라스의 옵티마이저를 임포트
import numpy as np # Numpy를 임포트
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dropout
import random
from tensorflow.python.keras.initializers import RandomUniform


pullData = open('RSIAUD.txt', 'r').read()
dataArray = pullData.split('\n')
xar =[]
yar=[]
qar =[]
war=[]
ear =[]
rar=[]
for eachLine in dataArray:
    if len(eachLine) >1:
        x1,x2,x3,x4,x5, x6, x7, x8, x9, x10, x11, x12, x13, y, z = eachLine.split(",")
        x1.append(float(x1))
        x2.append(float(x2))
        x3.append(float(x3))
        x4.append(float(x4))
        x5.append(float(x5))
        x6.append(float(x6))
        x7.append(float(x7))
        x8.append(float(x8))
        x9.append(float(x9))
        x10.append(float(x10))
        x11.append(float(x11))
        x12.append(float(x12))
        x13.append(float(x13))
        yar.append(float(y))
        zar.append(float(z))


print(xar)
Y=[]

for i in range (len(yar)):
    if yar[i] == 0:
        yar[i] = 0.5
    if yar[i] == -1:
        yar[i] = 0

for i in range(0,500):
    yar.insert(0,0)
    x1.insert(0,100)
    x2.insert(0,100)
    x3.insert(0,100)
    x4.insert(0,100)
    x5.insert(0,100)
    x6.insert(0, 100)
    x7.insert(0, 100)
    x8.insert(0, 100)
    x9.insert(0, 100)
    x10.insert(0, 100)
    x11.insert(0, 100)
    x12.insert(0, 100)
    x13.insert(0, 100)
    yar.insert(0,1)
    x1.insert(0,0)
    x2.insert(0,0)
    x3.insert(0,0)
    x4.insert(0,0)
    x5.insert(0,0)
    x6.insert(0, 0)
    x7.insert(0, 0)
    x8.insert(0, 0)
    x9.insert(0, 0)
    x10.insert(0, 0)
    x11.insert(0, 0)
    x12.insert(0, 0)
    x13.insert(0, 0)

feature =[]
feature.append([qar])
feature.append([war])
feature.append([ear])
feature.append([rar])
feature.append([xar])
print(feature[0])


print(yar)
TrainSet = [[] for j in range(3000)]

for i in range(0, 3000):
    TrainSet[i].append(qar[i])
    TrainSet[i].append(war[i])
    TrainSet[i].append(ear[i])
    TrainSet[i].append(rar[i])
    TrainSet[i].append(xar[i])

T_x = np.array(TrainSet).transpose()
print(T_x.shape)

X=np.array(xar[0:11000])
Q = np.array(qar[0:11000])
Q_val = np.array(qar[11000:20000])
X_val =np.array(xar[11000:20000])
Y_val =np.array(yar[11000:20000])
y=np.array(yar[0:11000]) #숫자 10부터 1
series = Series(X)
seriesq = Series(Q)
seriesqv = Series(Q_val)
seriesv = Series(X_val)
print(series)
# prepare data for normalization
values = series.values
valuesv = seriesv.values
valuesq = seriesq.values
valuesqv = seriesqv.values
values = values.reshape((len(values), 1))
valuesv = valuesv.reshape((len(valuesv), 1))
valuesq = valuesq.reshape((len(valuesq), 1))
valuesqv = valuesqv.reshape((len(valuesqv), 1))
# train the normalization
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(values)
normalized_VX = scaler.transform(valuesv)
normalized_Q = scaler.transform(valuesq)
normalized_VQ = scaler.transform(valuesqv)
normalized = scaler.transform(values)
# inverse transform and print
inversed = scaler.inverse_transform(normalized)

print(normalized.shape)
A = np.expand_dims(normalized,axis=0)
print(normalized.shape)

T = []
for i in range(0,11000):
    T.append((normalized[i], normalized_Q[i]))

V =[]
for i in range(0,5000):
    V.append((normalized_VX[i], normalized_VQ[i]))

S = np.array(T)
print(S.shape)

model=Sequential()
model.add(Dense(1,  kernel_initializer=RandomUniform(minval =1, maxval = 1), input_dim=2, activation='sigmoid'),)
sgd=optimizers.SGD(lr=0.01)
model.compile(optimizer=sgd ,loss='binary_crossentropy',metrics=['binary_accuracy'])
# 옵티마이저는 경사하강법의 일종인 확률적 경사 하강법 sgd를 사용합니다.
# 손실 함수(Loss function)는 binary_crossentropy(이진 크로스 엔트로피)를 사용합니다.
model.fit(S[:,:,0],y, batch_size=32, epochs=10, shuffle=False)
# 주어진 X와 y데이터에 대해서 오차를 최소화하는 작업을 200번 시도합니다.
M = model.predict(V[:,:,:0])

A = [0,10,20,30,40,50,60,70,80,90,100]
S = np.array(A)
S = S.reshape((len(S), 1))
print(model.predict(V[:,:,0]))

R =0
T =0
for i in range(0, len(M)):
    if M[i] > 0.7 and Y_val[i] != 0.5:
        T = T+1
        if Y_val[i] == 1:
            R = R+1
    if M[i]  < 0.3 and Y_val[i] != 0.5:
        T = T + 1
        if Y_val[i] == 0:
            R = R + 1
print(T)
print(R)
