import serial, time, io, math, sympy, mpmath, trilateration2D, random

import numpy as np
from sympy import Symbol, nsolve

from scipy.optimize import *
from trilateration2D import point,circle
import re

ser = serial.Serial()

ser.baudrate = 115200
ser.port = 'COM9'
ser.timeout = 0
ser.parity = 'N'
ser.stopbits = 1
ser.bytesize = 8
run = True

lengthInCM = 400 # Each side in CM
step = 41.0 # Defines the available steps accesible from the Microbit. -44 to -128 is the default
divisor = lengthInCM / step # Defines the cm per unit
noOfPlayers = 2 # Number of players available to tracking
playerMeasurement = []
player_list = [] # player


def playerList(player_list):
    for x in range(noOfPlayers):
        p = x+1
        player = 'P' + str(p)
        player_list.append(player)
        playerMeasurement.append([-1,-1])
playerList(player_list) #


A = [[0, 0],[lengthInCM, 0],[lengthInCM, lengthInCM],[0, lengthInCM]] # Defines coordinates of the Antennas
player_antenna_lists = [[[]for y in range(4)]for x in range(noOfPlayers)] # 3Dimensional list containing Recorded RSSI values for each Player-Antenna

test = "A1,RSSI:-80,MSG:P1"

x = Symbol('x')
y = Symbol('y')

# def myFunction(z):
#     x = z[0]
#     y = z[1]
#
#     F = [0,0]
#     F[0] = (x-0)**2 + (y-0)**2 - 2500
#     F[1] = (x-0)**2 + (y-50)**2 - 2500
#     return F
#
# zGuess = np.array([1,1])
# z = fsolve(myFunction,zGuess)
# print z

def checkDistanceArrays(): #Checking if there are at least 3 recorded distances
    checkList = [[]for x in range(noOfPlayers)]
    flags = [False for x in range(noOfPlayers)]
    for i in range(player_list.__len__()):
        if player_antenna_lists[i][0] != []:
            checkList[i].append(0)
        if player_antenna_lists[i][1] != []:
            checkList[i].append(1)
        if player_antenna_lists[i][2] != []:
            checkList[i].append(2)
        if player_antenna_lists[i][3] != []:
            checkList[i].append(3)

        if checkList[i].__len__() >= 3:
            flags[i] = True

    return [flags,checkList]


def trilaterate(distances,antennas):
    anchors = [0,0,0,0]

    for x in range(3):
        anchors[antennas[x]] = A[antennas[x]]
    anchors.remove(0)   # Deletes the remaining antenna




   # points = [trilateration2D.point(anchors[0][0],anchors[0][1]),trilateration2D.point(anchors[1][0], anchors[1][1]),trilateration2D.point()]
   #print points[0].x
   # circles = [trilateration2D.circle(points[0],distances[0]),trilateration2D.circle(points[1], distances[1]),trilateration2D.circle(points[2], distances[2])]
    #print circles[0].center

    points = [point(anchors[0][0], anchors[0][1]),point(anchors[1][0], anchors[1][1]),point(anchors[2][0], anchors[2][1])]


    if anchors[0] == [0,0]:
        for i in range(anchors.__len__()):
            flag = [False, False]
            for j in range(2):
                if anchors[i][j] != 0: #Check if referencepoint
                    flag[j] = True
            if flag[0] == True:
                points[i].x = distances[i]
            if flag[1] == True:
                points[i].y = distances[i]
    else:
        for i in range(anchors.__len__()):
            flag = [False, False]
            if anchors[i][0] != 400:# Check if referencepoint
                flag[0] = True
            if anchors[i][1] != 0:
                flag[1] = True
            if flag[0] == True:
                points[i].x = lengthInCM - distances[i]
            if flag[1] == True:
                points[i].y = distances[i]

    c1 = circle(points[0], distances[0])
    c2 = circle(points[1], distances[1])
    c3 = circle(points[2], distances[2])

    circles = [c1,c2,c3]


    inner_points = []
    for p in trilateration2D.get_all_intersecting_points(circles):
        if trilateration2D.is_contained_in_circles(p, circles):
            inner_points.append(p)
    if inner_points != []:
        center = trilateration2D.get_polygon_center(inner_points)
        return center
    # else:
    #     for i in range(points.__len__()):
    #         print points[i].x, points[i].y, "\n"
    #     return None
   #
   #
   #  #F[0] = (x - anchors[0][0]) ** 2 + (y - anchors[0][1]) ** 2 - length**2
   #
   #
   #
   #
   #  #circles = [[p1],[],[]]
   #  #print nsolve([((x-p1[0])**2 + (y-p1[1])**2) - d1**2, ((x-p2[0])**2 + (y-p2[1])**2) - d2**2 ], [x, y], [1, 1])
   #  # e1 = math.pow(,2)+math.pow
   #  # e2 = math.pow(x - p2[0], 2) + math.pow(y - p2[1], 2)
   #  # e3 = math.pow(x - p3[0], 2) + math.pow(y - p3[1], 2)

def translateDistToCoord():
    result = checkDistanceArrays()
    flags = result[0]
    checklist = result[1]

    for i in range(noOfPlayers):
        if flags[i] == True:
            d = [player_antenna_lists[i][checklist[i][0]][0],player_antenna_lists[i][checklist[i][1]][0],player_antenna_lists[i][checklist[i][2]][0]]
            center = trilaterate(d, checklist[i])
            if center != None:
                playerMeasurement[i] = [center.x,center.y]
            del player_antenna_lists[i][checklist[i][0]][0],player_antenna_lists[i][checklist[i][1]][0],player_antenna_lists[i][checklist[i][2]][0]



    #a[result[0]][0],a[result[1]][0],a[result[2]][0]


def convertSignalToCM(signal):
    rssi = math.fabs(signal) - 44
    result = rssi * divisor
    return round(result,2)


def extractRSSI(rssi):
    tmp = str(rssi).split(':')
    return float(tmp[1])
def extractPlayer(player):
    player = str(player).split(':')
    return player[1]
def checkSerialData(text):
    text = str(text)
    pattern = re.compile("A\d,RSSI:-\d\d,MSG:P\d")
    result = re.match(pattern, text)
    if result is not None:
        return [True,text]
    else:
        return [False]
def parse(text):
    text = str(text).split(',')
    tmp = convertSignalToCM(extractRSSI(text[1]))
    player = extractPlayer(text[2])

    pCount = 0
    for i in player_list:
        if player == i:
            if text[0] == 'A1':
                player_antenna_lists[pCount][0].append(tmp)
            elif text[0] == 'A2':
                player_antenna_lists[pCount][1].append(tmp)
            elif text[0] == 'A3':
                player_antenna_lists[pCount][2].append(tmp)
            elif text[0] == 'A4':
                player_antenna_lists[pCount][3].append(tmp)
        pCount +=1




direction = [[0,0],[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]]

weights = []
particle_set = []

ser.open()
while run:
    s = ser.read(18)
    result = checkSerialData(s)
    if result[0] is True:
        parse(result[1])
    time.sleep(0.2)
    translateDistToCoord()

    particle_filter(playerMeasurement)
    # for i in range(noOfPlayers):
    #     print "P", i + 1, ": X=", playerMeasurement[i][0], ", Y=", playerMeasurement[i][1]
ser.close()
test = "A1,RSSI:-80,MSG:P1"


def particle_filter(measurements):
    N = 250
    local_particle_set = [[random.randint(0,lengthInCM),random.randint(0,lengthInCM)]for x in range(N)]
    Z = measurements
    print local_particle_set
    calculateWeights(local_particle_set,Z)

def calculateWeights(set,Z):
    for i in set:
        i[0] = math.fabs(i[0]-Z[0])
        i[1] = math.fabs(i[1]-Z[1])




#particle_filter()