import serial, time, io, math, sympy, mpmath

from sympy import Symbol, nsolve


ser = serial.Serial()

ser.baudrate = 115200
ser.port = 'COM9'
ser.timeout = 0
ser.parity = 'N'
ser.stopbits = 1
ser.bytesize = 8
run = True

length = 500
step = 84.0
divisor = length / step

A = [[0, 0],[length, 0],[length, length],[0, length]]
a = [[] for y in range(4)]


test = "A1,RSSI:-80,MSG:P1"

x = Symbol('x')
y = Symbol('y')

def checkDistanceArrays():
    checkList = []
    if a[0] != []:
        checkList.append(0)
    if a[1] != []:
        checkList.append(1)
    if a[2] != []:
        checkList.append(2)
    if a[3] != []:
        checkList.append(3)

    if checkList.__len__() >= 3:
        return checkList
    else:
        return []

def trilaterate(d1, d2, d3,antennas):
    p1 = A[antennas[0]]
    p2 = A[antennas[1]]
    p3 = A[antennas[2]]

    print nsolve([((x-p1[0])**2 + (y-p1[1])**2) - d1**2, ((x-p2[0])**2 + (y-p2[1])**2) - d2**2, ((x-p3[0])**2 + (y-p3[1])**2) - d3**2], [x, y], [1, 1])

    # e1 = math.pow(,2)+math.pow



    # e2 = math.pow(x - p2[0], 2) + math.pow(y - p2[1], 2)
# e3 = math.pow(x - p3[0], 2) + math.pow(y - p3[1], 2)


def translateDistToCoord():
    result = checkDistanceArrays()
    if result != []:
        trilaterate(a[result[0]][0], a[result[1]][0], a[result[2]][0],result)
        del a[result[0]][0],a[result[1]][0],a[result[2]][0]
    else:
        print "Tom"


def convertSignalToCM(signal):
    rssi = math.fabs(signal) - 44
    result = rssi * divisor
    return result


def extractRSSI(rssi):
    tmp = str(rssi).split(':')
    return float(tmp[1])


def parse(text):
    text = str(text).split(',')
    tmp = convertSignalToCM(extractRSSI(text[1]))
    if text[0] == 'A1':
        a[0].append(tmp)
    elif text[0] == 'A2':
        a[1].append(tmp)
    elif text[0] == 'A3':
        a[2].append(tmp)
    elif text[0] == 'A4':
        a[3].append(tmp)





ser.open()
while run:
    s = ser.read(18)
    text = str(s)
    if text.__len__()>3:
        parse(text)
    time.sleep(0.4)
    translateDistToCoord()
ser.close()
