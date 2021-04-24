import matplotlib.pyplot as plt
pullData1 = open('BackTestLR55.txt', 'r').read()
dataArray1 = pullData1.split('\n')
xar1 =[]
yar1=[]
for eachLine in dataArray1:
    if len(eachLine) > 1:
        y1, x1, = eachLine.split(",")
        xar1.append(float(x1))
        yar1.append(float(y1))

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