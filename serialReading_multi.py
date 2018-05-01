import serial, time, io, math, trilateration2D, re

from random import *
from trilateration2D import point, circle
import matplotlib.pyplot as plt
from pylab import *


ser = serial.Serial()

ser.baudrate = 115200
ser.port = 'COM9'
ser.timeout = 0
ser.parity = 'N'
ser.stopbits = 1
ser.bytesize = 8
run = True

lengthInCM = 400  # Each side in CM
step = 26.0  # Defines the available steps accesible from the Microbit. -44 to -128 is the default
divisor = lengthInCM / step  # Defines the cm per unit
noOfPlayers = 2  # Number of players available to tracking
playerMeasurement = []
player_list = []  # player
direction = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]

class PlayerClass:
    def __init__(self):
        self.x = lengthInCM / 2.0
        self.y = lengthInCM / 2.0
        self.direction = direction[random.randint(0, direction.__len__())]
        self.movement_noise = 0.0

    def set(self, newX, newY, newDirection):
        if newX < 0 or newX >= lengthInCM:
            raise ValueError('X coordinate out of bound')
        if newX < 0 or newX >= lengthInCM:
            raise ValueError('Y coordinate out of bound')
        self.x = float(newX)
        self.y = float(newY)
        self.direction = direction[newDirection]

    def set_noise(self, newMovement_noise):
        self.movement_noise = float(newMovement_noise)


def playerList(player_list):
    for x in range(noOfPlayers):
        p = x + 1
        player = 'P' + str(p)
        player_list.append(player)
        playerMeasurement.append([-1, -1])


playerList(player_list)  #

A = [[0, 0], [lengthInCM, 0], [lengthInCM, lengthInCM], [0, lengthInCM]]  # Defines coordinates of the Antennas
player_antenna_lists = [[[] for y in range(4)] for x in
                        range(noOfPlayers)]  # 3Dimensional list containing Recorded RSSI values for each Player-Antenna
#player_antenna_lists[player][antenna][element] how to access
test = "A1,RSSI:-80,MSG:P1"



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

def checkDistanceArrays():  # Checking if there are at least 3 recorded distances
    checkList = [[] for x in range(noOfPlayers)]
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

    return [flags, checkList]


def trilaterate(distances, antennas):
    anchors = [0, 0, 0, 0]

    for x in range(3):
        anchors[antennas[x]] = A[antennas[x]]
    anchors.remove(0)  # Deletes the remaining antenna

    # points = [trilateration2D.point(anchors[0][0],anchors[0][1]),trilateration2D.point(anchors[1][0], anchors[1][1]),trilateration2D.point()]
    # print points[0].x
    # circles = [trilateration2D.circle(points[0],distances[0]),trilateration2D.circle(points[1], distances[1]),trilateration2D.circle(points[2], distances[2])]
    # print circles[0].center

    points = [point(anchors[0][0], anchors[0][1]), point(anchors[1][0], anchors[1][1]),
              point(anchors[2][0], anchors[2][1])]

    # if anchors[0] == [0, 0]: # Anchor at 0,0 is included
    for i in range(anchors.__len__()):
        flag = [False, False]
        for j in range(2):
            if anchors[i][j] != 0:  # Check if referencepoint
                flag[j] = True
        if flag[0] == True:
            points[i].x = distances[i]
        if flag[1] == True:
            points[i].y = distances[i]
    # else: # Anchor 0,0 is NOT included
    #     for i in range(anchors.__len__()):
    #         flag = [False, False]
    #         if anchors[i][0] != 400:  # Check if referencepoint
    #             flag[0] = True
    #         if anchors[i][1] != 0:
    #             flag[1] = True
    #         if flag[0] == True:
    #             points[i].x = lengthInCM - distances[i]
    #         if flag[1] == True:
    #             points[i].y = distances[i]

    c1 = circle(points[0], distances[0])
    c2 = circle(points[1], distances[1])
    c3 = circle(points[2], distances[2])

    circles = [c1, c2, c3]
    # print c1.center.x, c1.center.y, c2.center.x, c2.center.y, c3.center.x, c3.center.y
    # print c1.radius, c2.radius,c3.radius

    # clf()
    # plt.plot([0, lengthInCM, lengthInCM, 0], [0, 0, lengthInCM, lengthInCM], 'ro')
    # plt.plot([c1.center.x, c2.center.x, c3.center.x], [c1.center.y, c2.center.y, c3.center.y], 'bo')
    #
    # plt.pause(0.01)

    inner_points = []
    for p in trilateration2D.get_all_intersecting_points(circles):
        if trilateration2D.is_contained_in_circles(p, circles):
            #print p.x, p.y
            inner_points.append(p)
   # print inner_points
    time.sleep(1)
    if inner_points != []:
        center = trilateration2D.get_polygon_center(inner_points)

        return center


    # else:
    #     for i in range(points.__len__()):
    #         print points[i].x, points[i].y, "\n"
    #     return None
    return None

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
count = 1
countSuccess = 1
def translateDistToCoord():
    result = checkDistanceArrays()
    flags = result[0]
    checklist = result[1]
    global count, countSuccess
    for i in range(noOfPlayers):
        if flags[i]: # Check if there are at least 3 different different antenna values associated with player i
            d = [player_antenna_lists[i][checklist[i][0]][0], player_antenna_lists[i][checklist[i][1]][0],
                 player_antenna_lists[i][checklist[i][2]][0]]
            center = trilaterate(d, checklist[i])
            count += 1
            del player_antenna_lists[i][checklist[i][0]][0], player_antenna_lists[i][checklist[i][1]][0], \
                player_antenna_lists[i][checklist[i][2]][0]
            if center is not None:
                countSuccess += 1
                playerMeasurement[i] = [center.x, center.y]
                print playerMeasurement[i]
                return True
    return False

    # a[result[0]][0],a[result[1]][0],a[result[2]][0]


def convertSignalToCM(signal):
    rssi = math.fabs(signal) - 44
    result = rssi * divisor
    return round(result, 2)


def extractRSSI(rssi):
    tmp = str(rssi).split(':')
    return float(tmp[1])


def extractPlayer(player):
    player = str(player).split(':')

    return int(player[1].split('P')[1])-1


def checkSerialData(text):
    text = str(text)
    pattern = re.compile("A\d,RSSI:-\d\d,MSG:P\d")
    result = re.match(pattern, text)
    if result is not None:
        return [True, text]
    else:
        return [False]


def parse(text):
    text = str(text).split(',')
    antenna = int(text[0].split('A')[1])-1
    tmp = convertSignalToCM(extractRSSI(text[1]))
    player = extractPlayer(text[2])
    tmpList = player_antenna_lists[player][antenna]
    if(tmpList!=[]):
        #print "P",player," distance to A", antenna, "are: ", tmpList
        previousDistance = tmpList.pop()
        if(tmp != previousDistance):
           # print "Current: ",tmp,", Previous: ", previousDistance
            player_antenna_lists[player][antenna].append(previousDistance)
            player_antenna_lists[player][antenna].append(tmp)
    else:
        player_antenna_lists[player][antenna].append(tmp)
    # pCount = 0
    # for i in player_list:
    #     if player == i:
    #         if text[0] == 'A1':
    #             player_antenna_lists[pCount][0].append(tmp)
    #         elif text[0] == 'A2':
    #             player_antenna_lists[pCount][1].append(tmp)
    #         elif text[0] == 'A3':
    #             player_antenna_lists[pCount][2].append(tmp)
    #         elif text[0] == 'A4':
    #             player_antenna_lists[pCount][3].append(tmp)
    #     pCount += 1


def addRandomMotion(set):
    for i in range(set.__len__()):
        calc = set[i][0] + set[i][2][0] * randint(1, 10) * divisor
        if calc < 0:
            set[i][0] = 0
        elif calc > lengthInCM:
            set[i][0] = lengthInCM
        else:
            set[i][0] = calc
        calc = set[i][1] + set[i][2][1] * randint(1, 10) * divisor
        if calc < 0:
            set[i][1] = 0
        elif calc > lengthInCM:
            set[i][1] = lengthInCM
        else:
            set[i][1] = calc
    return set

weights = []
particle_set = []
def particle_filter(measurement):
    N = 250
    particle_set = [[randint(0, lengthInCM), randint(0, lengthInCM),
                           direction[randint(0, direction.__len__()-1)]] for x in range(N)]
    local_particle_set = particle_set
    Z = measurement

    #Predict
    local_particle_set = addRandomMotion(local_particle_set)

    # calculateWeights(local_particle_set,Z)
#particle_filter([])

def calculateWeights(set, Z):
    for i in set:
        i[0] = math.fabs(i[0] - Z[0])
        i[1] = math.fabs(i[1] - Z[1])


def gaussian(mu, sigma, x):
    # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
    return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))


def measurement_prob(p, measurement):
    prob = 1.0
    for i in range(4):
        dist = sqrt((p.x - A[i][0]) ** 2 + (p.y - A[i][1]) ** 2)
        prob *= gaussian(dist, 0.05, measurement[i])
    return prob
# fig = plt.figure(1)
# plt.plot([0],[0],'ro')
# fig = plt.figure(2)
# plt.plot([lengthInCM],[0],'ro')
# fig = plt.figure(3)
# plt.plot([lengthInCM],[lengthInCM],'ro')
# fig = plt.figure(4)
# plt.plot([0],[lengthInCM],'ro')


# x = np.linspace(0, 10*np.pi, 100)
# y = np.sin(x)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot([0],[0],'ro')
# plt.ion()
#
# line1, = ax.plot(x, y, 'b-')
# for phase in np.linspace(0, 10*np.pi, 100):
#     line1.set_ydata(np.sin(0.5 * x + phase))
#     fig.canvas.draw()




# add this if you don't want the window to disappear at the end

# plt.plot(100, 100, 'bo')
# fig.add_subplot()
#plt.bar(player_list,10)


#plt.plot([0, lengthInCM, lengthInCM, 0], [0,0 , lengthInCM, lengthInCM], 'ro')


plt.ion()

ser.open()

while run:
    s = ser.read(18)
    result = checkSerialData(s)
    if result[0] is True:
        parse(result[1])
        #print count, countSuccess, float(countSuccess)/count
    time.sleep(0.01)
    if translateDistToCoord() is True:
        clf()
        plt.plot([0, lengthInCM, lengthInCM, 0], [0, 0, lengthInCM, lengthInCM], 'ro')
        plt.plot(playerMeasurement[0][0], playerMeasurement[0][1], 'bo')
        plt.plot(playerMeasurement[1][0], playerMeasurement[1][1], 'g+')
        plt.pause(0.01)

    #particle_filter(playerMeasurement)

    # for i in range(noOfPlayers):
    #     print "P", i + 1, ": X=", playerMeasurement[i][0], ", Y=", playerMeasurement[i][1]
ser.close()
plt.show()
# test = "A1,RSSI:-80,MSG:P1"








# particle_filter()
