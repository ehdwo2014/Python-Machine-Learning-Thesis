import matplotlib.pyplot as plt
pullData1 = open('BackTestLR54.txt', 'r').read()
dataArray1 = pullData1.split('\n')
xar1 =[]
yar1=[]
for eachLine in dataArray1:
    if len(eachLine) > 1:
        y1, x1, = eachLine.split(",")
        xar1.append(float(x1))
        yar1.append(float(y1))

A =[]
Dif =[]
init = 0
init = yar1[0]
InitialBudget = 100000

for i in range(0, len(yar1)):
    yar1[i] = yar1[i] - i * 12


Lot_Size = (30000)/(yar1[len(yar1)-1])

for i in range(0, len(yar1)):
    yar1[i] = InitialBudget + (yar1[i] * Lot_Size)


print(yar1[len(yar1)-1])


def max_drawdown(X):
    mdd = 0
    peak = X[0]
    for x in X:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > mdd:
            mdd = dd
    return mdd

drawSeries = max_drawdown(yar1)
MaxDD = abs((drawSeries))

gainGross =0
lossGross =0


for i in range(0, len(yar1) - 1):
    if yar1[i] - yar1[i+1] > 0:
        lossGross =  (yar1[i+1]-yar1[i]) + lossGross
    if yar1[i] - yar1[i+1] < 0:
        gainGross = (yar1[i+1]-yar1[i]) + gainGross

print(gainGross)
print(lossGross)
print(abs(gainGross/lossGross))
print(Lot_Size)
print(MaxDD * 100)

print(len(yar1))