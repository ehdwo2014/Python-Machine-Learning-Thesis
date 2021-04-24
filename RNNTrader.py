import numpy as np
from time import sleep

from keras.models import load_model
model = load_model('my_model.h5')
while True:
    f = open('RawData.txt', 'r')
    dataArray = f.read().split('\n')
    x1ar = []
    x2ar = []
    x3ar = []
    x4ar = []
    x5ar = []
    x6ar = []
    x7ar = []
    x8ar = []
    x9ar = []
    x10ar = []
    x11ar = []
    x12ar = []
    x13ar = []

    for eachLine in dataArray:
        if len(eachLine) > 1:
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13 = eachLine.split(",")
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
    Input = [[] for j in range(1)]
    for i in range(0, len(x1ar)):
        Input[i].append(x1ar[i] * 0.01)
        Input[i].append(x2ar[i] * 0.01)
        Input[i].append(x3ar[i] * 0.01)
        Input[i].append(x4ar[i] * 0.01)
        Input[i].append(x5ar[i] * 0.01)
        Input[i].append(x6ar[i] * 0.01)
        Input[i].append(x7ar[i] * 0.01)
        Input[i].append(x8ar[i] * 0.01)
        Input[i].append(x9ar[i] * 0.01)
        Input[i].append(x10ar[i] * 0.01)
        Input[i].append(x11ar[i] * 0.01)
        Input[i].append(x12ar[i] * 0.01)
        Input[i].append(x13ar[i] * 0.01)
    if(len(x1ar) > 0):
        S = np.array(Input)
        MP = model.predict(S)
        print(MP)
        f.close()
        print(x2ar)
        try:
            A = open(r"C:\Users\DongJae Yoon\AppData\Roaming\MetaQuotes\Terminal\Common\Files\Signal.txt", 'w+')
            A.write(str(MP[0][0]))
            A.close()
        except Exception:
            pass
    x1ar.clear()
    x2ar.clear()
    x3ar.clear()
    x4ar.clear()
    x5ar.clear()
    x6ar.clear()
    x7ar.clear()
    x8ar.clear()
    x9ar.clear()
    x10ar.clear()
    x11ar.clear()
    x12ar.clear()
    x13ar.clear()
    Input.clear()
    dataArray.clear()
    sleep(0.5)