import serial, time, io, math, trilateration2D, re, thread

from random import *
from trilateration2D import point, circle
import matplotlib.pyplot as plt
from pylab import *


side_X_inCM = 372 # Width in CM
side_Y_inCM = 310 # Height in CM
# weakestRSSI = 84
# step = 84-(128-weakestRSSI)  # Defines the available steps accesible from the Microbit. -44 to -128 is the default
weakestRSSI = 84
step = weakestRSSI - 44.0

EMWAweight = 0.5

lengthPerUnitX = float(side_X_inCM/step)
lengthPerUnitY = float(side_Y_inCM/step)
maxDistance = sqrt(side_X_inCM**2+side_Y_inCM**2)
noOfPlayers = 2  # Number of players available to tracking
playerMeasurement = []
fieldDirection  = 174


directionVectors = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
degreeDirection= [[90,90],[270,270],[0,0],[180,180],[1,89],[91,179],[271,359],[181,269]]

class PlayerClass:
    def __init__(self,id,x,y,movement_noise,direction):
        self.id = id
        self.x = x
        self.y = y
        self.direction = direction
        self.movement_noise = movement_noise
        self.DistanceToAntennas = [[]for i in range(4)]
        self.p_set = []

    def set(self, newX, newY, newDirection):
        if newX < 0:
            self.x = 0
        elif newX >= side_X_inCM:
            self.x = side_X_inCM
        else:
            self.x = float(newX)
        if newY < 0:
            self.y = 0
        elif newY >= side_Y_inCM:
            self.y = side_Y_inCM
        else:
            self.y = float(newY)

        self.newDirection(newDirection)
    def set_p_set(self, set):
        self.p_set = set
    def set_noise(self, newMovement_noise):
        self.movement_noise = float(newMovement_noise)
    def addDistanceToAntenna(self,Distance,antennaId):
        self.DistanceToAntennas[antennaId].append(Distance)
    def newDirection(self,direction):
        self.direction = direction
    def checkRecordedDistances(self):  # Checking if there are at least 3 recorded distances
        checkList = []
        for i in range(4):
            if self.DistanceToAntennas[i] != []:
                checkList.append(i)
        return checkList

    def __str__(self):
        return "Player:" + str(self.id+1) + " X=" + str(self.x) + ",Y=" + str(self.y) + ",Direction="+ str(self.direction) + ",Movement_Noise=" + str(self.movement_noise)
    def print_Set(self):
        print self.p_set





class AntennaClass:
    def __init__(self,id,x,y):
        self.x = x
        self.y = y
        self.id = id
    def __str__(self):
        return "Antenna:" + str(self.id+1) + " X=" + str(self.x) + ",Y=" + str(self.y)

N = 200
particleSets = []

def generateParticleSets(direction):
    for j in range(noOfPlayers):
        particleSets.append([])
        for i in range(N):
            particleSets[j].append(PlayerClass(i,randint(0, side_X_inCM),randint(0, side_Y_inCM),0.7,direction))
def generateParticleSet():
    p_set = []
    for i in range(N):
        p_set.append(PlayerClass(i,randint(0, side_X_inCM),randint(0, side_Y_inCM),0.05,randint(0,359)))
    return p_set



def convertDegreeToVector(degree):
    dirCount = 1
    for set in degreeDirection:
        if set[0] <= degree and set[1] >= degree:
            break
        dirCount += 1
    return directionVectors[dirCount]







def createPlayerList(): # Generates a set over registered players
    list = []  # p
    for x in range(noOfPlayers):
        list.append(PlayerClass(x,side_X_inCM/2.0,side_Y_inCM/2.0,0.7,0))
        list[x].set_p_set(generateParticleSet())
    return list
playerList = createPlayerList()

def printPlayers():
    for i in range(playerList.__len__()):
        print playerList[i]


def createAntennaList(): # Generates a set over registered antennas
    list = []
    list.append(AntennaClass(0,0,0))
    list.append(AntennaClass(1,side_X_inCM,0))
    list.append(AntennaClass(2,side_X_inCM,side_Y_inCM))
    list.append(AntennaClass(3,0,side_Y_inCM))
    return list
antennaList = createAntennaList()

def printAntennas():
    for i in range(antennaList.__len__()):
        print antennaList[i]






test = "A1,RSSI:-80,MSG:P1"
rssiSum = 0
counter = 0
min = 0
def convertSignalToCM(signal):
    # Warmup to find weakest value for the calibrationprocess
    # global rssiSum, counter, min
    # rssiSum += signal
    # counter+=1.0
    # if signal < min:
    #     min = signal
    # print "Average: ", rssiSum / counter, ", Min: ", min
    # counter += 1.0

    #n = 5;
    #r = r0- 10 * n * math.log(d,10)

    rssi = math.fabs(signal) - 44
    procent = float(rssi/(weakestRSSI-44))
    # print "% = ", procent
    #
    # print "LPX: ", lengthPerUnitX, ", LPY: " , lengthPerUnitY
    # print "Signal: ", rssi, ", RSSI*X: ", rssi*lengthPerUnitX, ", RSSI*Y: ", rssi*lengthPerUnitY
    # print "% of X: ", side_X_inCM*procent,", % of Y: " , side_Y_inCM*procent
    #
    # #result = sqrt((side_X_inCM*procent)**2+(side_Y_inCM*procent)**2)
    # result2 = sqrt((rssi*lengthPerUnitX)**2+(rssi*lengthPerUnitY)**2)
    # print "RSSI * Length: ", result2

    result = procent*maxDistance
    # if result> side_X_inCM:
    #     result = side_X_inCM
    #print "Distance: ", result
    #result = sqrt((rssi*lengthPerUnitX*procent) ** 2 + (rssi*lengthPerUnitY*procent) ** 2)
    #result = rssi * scalingUnit
   # print result
    return round(result, 2)


def extractRSSI(rssi):
    return float(rssi)




def checkSerialData(text):
    text = str(text)
    pattern = re.compile("A\d,RSSI:-\d\d,MSG:P\d")
    result = re.match(pattern, text)
    if result is not None:
        return [True, text]
    else:
        return [False]

RSSI_Antenna = [[] for x in range(4)]
def parse(text):
    try:
        text = str(text).split(',')
        antenna = int(text[0])-1
        convertedRSSI = convertSignalToCM(extractRSSI(text[1]))
        player = int(text[2])
        direction = int(text[3])
        timestamp = int(text[4])
    except (IndexError, ValueError):
        print "InvalidData"
        gateway.reset_input_buffer()
        return
    RSSI_Antenna[antenna].append(int(text[1]))
    playerList[player].direction = direction
    distanceArray = playerList[player].DistanceToAntennas[antenna]
    if distanceArray != []:
        #print "P",player," distance to A", antenna, "are: ", tmpList
        previousDistance = playerList[player].DistanceToAntennas[antenna][distanceArray.__len__()-1]
        if convertedRSSI != previousDistance:
           # print "Current: ",tmp,", Previous: ", previousDistance
           playerList[player].DistanceToAntennas[antenna].append(convertedRSSI)
    else:
        playerList[player].DistanceToAntennas[antenna].append(convertedRSSI)



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
                #particle_filter(center,i,checklist[i])
                countSuccess += 1
                playerMeasurement[i] = [center.x, center.y]
                print playerMeasurement[i]
                return True
    return False

    # a[result[0]][0],a[result[1]][0],a[result[2]][0]



def trilaterate2(RSSI_values):
    points = []
    anchors = []

    distances = []
    offsetFactor = 3.0
    for RSSI in RSSI_values:
        distances.append(convertSignalToCM(RSSI[0]))


    points.append(point(distances[0]-distances[0]/offsetFactor,distances[0]-distances[0]/offsetFactor))
    points.append(point(side_X_inCM-distances[1] + distances[1] / offsetFactor, distances[1] - distances[1] /offsetFactor))
    points.append(point(side_X_inCM-distances[2] + distances[2]/ offsetFactor, side_Y_inCM-distances[2] + distances[2] / offsetFactor))
    points.append(point(distances[3] - distances[3]/ offsetFactor, side_Y_inCM-distances[3] + distances[3] / offsetFactor))
    maxD = 0
    for i in range(3):
        for j in range(i+1,4,1):
            d = trilateration2D.get_two_points_distance(points[i],points[j])
            if d > maxD:
                maxD = d
    print "MAXD: " +  str(maxD)

    # points.append(point(0,0))
    # points.append(point(side_X_inCM,0))
    # points.append(point(side_X_inCM,side_Y_inCM))
    # points.append(point(0,side_Y_inCM))


    circles = []
    for i in range(4):
        circles.append(circle(points[i], maxD))

    inner_points = []
    for p in trilateration2D.get_all_intersecting_points(circles):
        if trilateration2D.is_contained_in_circles(p, circles):
            inner_points.append(p)
        # print p.x, p.y
    if inner_points != []:
        center = trilateration2D.get_polygon_center(inner_points)
        print center.x,center.y

        clf()
        plt.plot([0, side_X_inCM, side_X_inCM, 0], [0, 0, side_Y_inCM, side_Y_inCM], 'ro')
        plt.plot(center.x, center.y, 'bo')
        plt.plot(side_X_inCM/2, side_Y_inCM/2, 'g+')
        plt.plot(xAvg,yAvg,'y*')
        plt.pause(0.00000000001)



def trilaterate(distances, antennas):
    anchors = []

    for x in range(3):
        anchors.append(antennaList[antennas[x]])
    offsetFactor = 2.0
    points = []



    # for anchor in anchors:
    #     points.append(point(anchor.x,anchor.y))
    # for anchor in anchors:
    #     #print anchor.id
    #     if anchor.id == 0:
    #         points.append(point(distances[0][0]-distances[0][0]/offsetFactor,distances[0][0]-distances[0][0]/offsetFactor))
    #     elif anchor.id == 1:
    #         points.append(point(side_X_inCM-distances[1][0] + distances[1][0] / offsetFactor, distances[1][0] - distances[1][0] /offsetFactor))
    #     elif anchor.id == 2:
    #         points.append(point(side_X_inCM-distances[2][0] + distances[2][0] / offsetFactor, side_Y_inCM-distances[2][0] + distances[2][0] / offsetFactor))
    #     else:
    #         points.append(point(distances[3][0] - distances[3][0]/ offsetFactor, side_Y_inCM-distances[3][0] + distances[3][0] / offsetFactor))
    #

    # for anchor in anchors:
    #     #print anchor.id
    #     if anchor.id == 0:
    #         points.append(point(distances[0][0]+distances[0][0]/offsetFactor,distances[0][0]+distances[0][0]/offsetFactor))
    #     elif anchor.id == 1:
    #         points.append(point(side_X_inCM-distances[1][0] - distances[1][0] / offsetFactor, distances[1][0] + distances[1][0] /offsetFactor))
    #     elif anchor.id == 2:
    #         points.append(point(side_X_inCM-distances[2][0] - distances[2][0] / offsetFactor, side_Y_inCM-distances[2][0] - distances[2][0] / offsetFactor))
    #     else:
    #         points.append(point(distances[3][0] + distances[3][0]/ offsetFactor, side_Y_inCM-distances[3][0] - distances[3][0] / offsetFactor))

    for anchor in anchors:
        # print anchor.id
        if anchor.id == 0:
            points.append(point(0,0))
        elif anchor.id == 1:
            points.append(point(side_X_inCM,0))
        elif anchor.id == 2:
            points.append(point(side_X_inCM,side_Y_inCM))
        else:
            points.append(point(0,side_Y_inCM))


    for p in points:
        if p.x > side_X_inCM:
            p.x = side_X_inCM
        elif p.x < 0:
            point.x = 0
        if p.y > side_Y_inCM:
            p.y = side_Y_inCM
        elif p.y < 0:
            p.y  = 0
    # points = [point(anchors[0].x, anchors[0].y), point(anchors[1].x, anchors[1].y),
    #           point(anchors[2].x, anchors[2].y)]
    # print points
    # if anchors[0] == [0, 0]: # Anchor at 0,0 is included
    # for i in range(anchors.__len__()):
    #     flag = [False, False]
    #     for j in range(2):
    #         if anchors[i][j] != 0:  # Check if referencepoint
    #             flag[j] = True
    #     if flag[0] == True:
    #         points[i].x = distances[i]
    #     if flag[1] == True:
    #         points[i].y = distances[i]
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


    circles = []
    for i in range(3):
        d = distances[antennas[i]][0]
        if d == 0:
            return point(anchors[i].x,anchors[i].y)
        circles.append(circle(points[i], d))
    #     print "Antenna: ", anchors[i].id+1, " with distance: ", distances[antennas[i]][0], " became -> "
    #     print  circles[i].center.x, circles[i].center.y, circles[i].radius
    # print maxDistance
    # c1 = circle(points[0], distances[antennas[0]][0]+sqrt(lengthPerUnitX**2+lengthPerUnitY**2))
    # c2 = circle(points[1], distances[antennas[1]][0]+sqrt(lengthPerUnitX**2+lengthPerUnitY**2))
    # c3 = circle(points[2], distances[antennas[2]][0]+sqrt(lengthPerUnitX**2+lengthPerUnitY**2))

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
# #
# #
# #  #F[0] = (x - anchors[0][0]) ** 2 + (y - anchors[0][1]) ** 2 - length**2
# #
# #
# #
# #
# #  #circles = [[p1],[],[]]
# #  #print nsolve([((x-p1[0])**2 + (y-p1[1])**2) - d1**2, ((x-p2[0])**2 + (y-p2[1])**2) - d2**2 ], [x, y], [1, 1])
# #  # e1 = math.pow(,2)+math.pow
# #  # e2 = math.pow(x - p2[0], 2) + math.pow(y - p2[1], 2)
# #  # e3 = math.pow(x - p3[0], 2) + math.pow(y - p3[1], 2)
# count = 1
# countSuccess = 1


#

#
#

# # fig = plt.figure(1)
# # plt.plot([0],[0],'ro')
# # fig = plt.figure(2)
# # plt.plot([lengthInCM],[0],'ro')
# # fig = plt.figure(3)
# # plt.plot([lengthInCM],[lengthInCM],'ro')
# # fig = plt.figure(4)
# # plt.plot([0],[lengthInCM],'ro')
#
#
# # x = np.linspace(0, 10*np.pi, 100)
# # y = np.sin(x)
# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # plt.plot([0],[0],'ro')
# # plt.ion()
# #
# # line1, = ax.plot(x, y, 'b-')
# # for phase in np.linspace(0, 10*np.pi, 100):
# #     line1.set_ydata(np.sin(0.5 * x + phase))
# #     fig.canvas.draw()
#
#
#
#
# # add this if you don't want the window to disappear at the end
#
# # plt.plot(100, 100, 'bo')
# # fig.add_subplot()
# #plt.bar(player_list,10)
#
#
# #plt.plot([0, lengthInCM, lengthInCM, 0], [0,0 , lengthInCM, lengthInCM], 'ro')
#
#
# plt.ion()
#


r = sqrt(1 ** 2 + 1 ** 2)
x = cos(3*pi/2) * r
y = sin(0) * r

deg = degrees((3*pi/2))
rad = radians(pi)


def addRandomMotion(particleSet):

    # print particleSet[0]
    # for i in range(particleSet.__len__()):
    #     # vector = convertDegreeToVector(particleSet[i].direction)
    #     calc = particleSet[i].x + vector[0] * randint(1, 10)
    #     if calc < 0:
    #         particleSet[i].x = 0
    #     elif calc > side_X_inCM:
    #         particleSet[i].x = side_X_inCM
    #     else:
    #         particleSet[i].x = calc
    #     calc = particleSet[i].y + vector[1] * randint(1, 10)
    #     if calc < 0:
    #         particleSet[i].y = 0
    #     elif calc > side_Y_inCM:
    #         particleSet[i].y = side_Y_inCM
    #     else:
    #         particleSet[i].y = calc
    # print particleSet[0]

    # print particleSet[0]
    # for i in range(particleSet.__len__()):
    #     # vector = convertDegreeToVector(particleSet[i].direction)
    #     r = sqrt(particleSet[i].x**2 + particleSet[i].y**2)
    #     x = cos(particleSet[i].direction)*r
    #     y = sin(particleSet[i].direction)*r
    #
    #     calc = particleSet[i].x + vector[0] * randint(1, 10)
    #     if calc < 0:
    #         particleSet[i].x = 0
    #     elif calc > side_X_inCM:
    #         particleSet[i].x = side_X_inCM
    #     else:
    #         particleSet[i].x = calc
    #     calc = particleSet[i].y + vector[1] * randint(1, 10)
    #     if calc < 0:
    #         particleSet[i].y = 0
    #     elif calc > side_Y_inCM:
    #         particleSet[i].y = side_Y_inCM
    #     else:
    #         particleSet[i].y = calc
    print particleSet[0]

# def calculateWeights(set, Z):
#     for i in set:
#         i[0] = math.fabs(i[0] - Z[0])
#         i[1] = math.fabs(i[1] - Z[1])
#
#
def gaussian(mu, sigma, x):
    # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
    return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))
#
#
def measurement_prob(particle, measurement,checkList):
    prob = 1.0
    for i in range(3):
        dist = sqrt((particle.x - antennaList[checkList[i]].x) ** 2 + (particle.y - antennaList[checkList[i]].y) ** 2)
        #print dist,measurement[checkList[i]], gaussian(400, 5, 200)
        #time.sleep(1)
        prob *= gaussian(dist, 50 , measurement[i])
        #print prob
    return prob
def particle_filter(player, checkList):


    # z = []
    # for i in range(3):
    #     #print antennaList[checkList[i]]
    #     dist = sqrt((player.x - antennaList[checkList[i]].x) ** 2 + (player.y - antennaList[checkList[i]].y) ** 2)
    #     print dist, player.DistanceToAntennas[checkList[i]]
    #     #print "Innan Gauss", dist
    #     dist += gauss(0.0, 5)
    #     #print "Efter gauss", dist
    #     z.append(dist)
    # addRandomMotion(player.direction, player.p_set)

    z = []
    for i in range(3):
        # print antennaList[checkList[i]]
        dist = player.DistanceToAntennas[checkList[i]][0]
        # print "Innan Gauss", dist
        dist += gauss(0.0, 5)
        # print "Efter gauss", dist
        z.append(dist)
    addRandomMotion(player.p_set)


    w = []
    #calculateWeights
    for i in range(N):
       w.append(measurement_prob(player.p_set[i],z,checkList))
    normW = []
    normSum = sum(w)
    #print normSum
    #Normalize weights
    for i in range(N):
       normW.append(w[i]/normSum)
    #print normW
    #print "----"
    # time.sleep(0.1)
    # #Resample
    p3 = []

    index = randint(0,N-1)
    beta = 0.0
    mw = max(normW)
    #print "MW: ",mw
    for i in range(N):
        beta += random() * 2.0 * mw
        while beta > normW[index]:
            beta -= normW[index]
            index = (index + 1) % N
        p3.append(player.p_set[index])
    for p in p3:
        print p
        #time.sleep(0.1)
    player.set_p_set(p3)
    raw_input("Press Enter to continue...")
    print "FINISHED RUN--------------------"

    # for i in range(10):
    #     # Predict
    #     addRandomMotion(player.direction,particleSets[player.id])
    #     for i in range(N):
    #         particleSets[player.id][i].newDirection()
    # #
    #
    # w = []
    # #calculateWeights
    # for i in range(N):
    #    w.append(measurement_prob(particleSets[player.id][i],z,checkList))
    # #print w
    # normW = []
    # normSum = sum(w)
    # #print normSum
    # #Normalize weights
    # for i in range(N):
    #    normW.append(w[i]/normSum)
    # #print normW
    # #print "----"
    # time.sleep(0.1)
    # #Resample
    # p3 = []
    #
    # index = randint(0,N-1)
    # beta = 0.0
    # mw = max(normW)
    # #print "MW: ",mw
    # for i in range(N):
    #     beta += random() * 2.0 * mw
    #     while beta > normW[index]:
    #         beta -= normW[index]
    #         index = (index + 1) % N
    #     p3.append(particleSets[player.id][index])
    # for p in p3:
    #     print p
    #     #time.sleep(0.1)
    # particleSets[player.id]=p3
    # print "FINISHED RUN--------------------"



run = True


gateway = serial.Serial()
gateway.baudrate = 115200
gateway.timeout = None
gateway.parity = 'E'
gateway.port = "COM9"

gateway.open()







xAvg = 0; yAvg = 0; xMin = side_X_inCM; yMin = side_Y_inCM; xMax = 0;yMax = 0;
xSum = 0; ySum = 0



buffert = []
currentTimestamp = -1;
tsCounter = 0;

standardAvvikelseX = 0
standardAvvikelseY = 0
plt.ion()
# 1,-68,1,7,154

prevCounter = 0
def storeInBuffert(msg):
    global tsCounter,currentTimestamp, buffert, prevCounter
    try:
        text = str(msg).split(',')
        timestamp = int(text[4])
        if tsCounter == 0:
            antenna = int(text[0])-1
            convertedRSSI = extractRSSI(text[1])
            player = int(text[2])
            direction = int(text[3])
            buffert.append([antenna,convertedRSSI,player,direction,timestamp])
            tsCounter+=1
            currentTimestamp = timestamp
        else:
            if timestamp == currentTimestamp:
                antenna = int(text[0]) - 1
                convertedRSSI = extractRSSI(text[1])
                player = int(text[2])
                direction = int(text[3])
                buffert.append([antenna, convertedRSSI, player, direction, timestamp])
                tsCounter += 1
            else:
                # Discard the current buffer
                buffert = []
                antenna = int(text[0]) - 1
                convertedRSSI = extractRSSI(text[1])
                player = int(text[2])
                direction = int(text[3])
                buffert.append([antenna, convertedRSSI, player, direction, timestamp])
                # wastedPackets = tsCounter
                tsCounter = 1
                currentTimestamp = timestamp
                print "Discarded the buffer"
                # return wastedPackets
    except (IndexError, ValueError):
        print "InvalidData"
        gateway.reset_input_buffer()
        return
    if tsCounter == 4:
        #print buffert
        # print "Write to Permanent Storage"
        if RSSI_Antenna[0].__len__() == 0:
            for x in range(4):
                RSSI_Antenna[x].append(buffert[x][1]*(1-EMWAweight))
        else:
            for x in range(4):
                prevVal = RSSI_Antenna[x][0]
                v = EMWAweight*prevVal+(1-EMWAweight)*buffert[x][1]
                print "V: " + str(v)
                RSSI_Antenna[x].pop()
                RSSI_Antenna[x].append(v)
            prevCounter += 1
        tsCounter = 0;


        # trilaterate2(RSSI_Antenna)
        # for x in range(4):
        #     RSSI_Antenna[x].pop()
        del buffert
        buffert = []
        #raw_input("Press space to continue...")

seconds = 15*5
runTimes = seconds

def countDownClock():
    t = seconds
    global run
    while t > 0:
        time.sleep(1)
        t-=1
    run = False

# thread.start_new(countDownClock,()) # Trad som kor programmet i en angiven tid.
while run:



    # runTimes-=1
    # if(runTimes==0):
    #     run = False


    starttime = time.time();
    serialData = gateway.readline()
    serialData = str(serialData).strip('\n')
    #now = time.time()
    #elapsed  = now - starttime
    # xSum +=elapsed
    # counter+=1.0
    #print elapsed
    # runTimes-=1
    # if(runTimes==0):
    #     run = False

    print "S: ", serialData
    storeInBuffert(serialData)
    #result = checkSerialData(s)
    #if result[0] is True:
    # parse(serialData)
        #print result[1]
    #time.sleep(0.01) # Tested different values to find optimal wait-value for ser.read() to be able to read in time.

    # for player in playerList: # Check each player if coordinates can be detetermined
    #     checkList = player.checkRecordedDistances()
    #     if checkList.__len__()>=3:
    #         #print "Trying to trilaterate: "
    #         #print "Px: ", player.x, ", Py: ", player.y
    #         center = trilaterate(player.DistanceToAntennas,checkList)
    #         if center is not None:
    #
    #             #Check the avg of trilaterated values
    #             xSum += center.x; ySum += center.y;
    #             if center.x > xMax:
    #                 xMax = center.x
    #             if center.y > yMax:
    #                 yMax = center.y
    #             counter += 1
    #             xAvg = xSum/counter
    #             yAvg = ySum /counter
    #
    #
    #             standardAvvikelseX+= (center.x - side_X_inCM/2)**2
    #             standardAvvikelseY += (center.y - side_Y_inCM / 2)**2

                # clf()
                # plt.plot([0, side_X_inCM, side_X_inCM, 0], [0, 0, side_Y_inCM, side_Y_inCM], 'ro')
                # plt.plot(center.x, center.y, 'bo')
                # plt.plot(side_X_inCM/2, side_Y_inCM/2, 'g+')
                # plt.plot(xAvg,yAvg,'y*')
                # plt.pause(0.00000000001)
                # player.set(center.x,center.y,player.direction)
                # print "Trilaterade values: " , center.x, center.y, ", Direction: ", player.direction

    #         #     runTimes-=1
    #         #     if(runTimes==0):
    #         #         run = False
    #         particle_filter(player, checkList)
    #         for p in player.p_set:
    #             print p
    #         raw_input("Press Space to continue...")
    #
    #
    #
    #         player.DistanceToAntennas[checkList[0]].pop(0)
    #         player.DistanceToAntennas[checkList[1]].pop(0)
    #         player.DistanceToAntennas[checkList[2]].pop(0)

            #print "Player Distance-Antennas after 3 uniqe are determined", player.DistanceToAntennas
            #particle_filter(player,checkList)

    # if translateDistToCoord() is True:
    #     clf()
    #     plt.plot([0, lengthInCM, lengthInCM, 0], [0, 0, lengthInCM, lengthInCM], 'ro')
    #     plt.plot(playerMeasurement[0][0], playerMeasurement[0][1], 'bo')
    #     plt.plot(playerMeasurement[1][0], playerMeasurement[1][1], 'g+')
    #     plt.pause(0.01)

    #particle_filter(playerMeasurement)

    # for i in range(noOfPlayers):
    #     print "P", i + 1, ": X=", playerMeasurement[i][0], ", Y=", playerMeasurement[i][1]
gateway.close()
# data_file_antennas = []
# for x in range(4):
#     data_file_antennas.append(open("rssiData_A"+str(x+1)+"_EWMA","w"))
# count = 0
# # for antenna in RSSI_Antenna:
# #     print antenna
# for antenna in RSSI_Antenna:
#     data_file_antennas[count].write(str(antenna))
#     count+=1
# for x in range(4):
#     data_file_antennas[x].close()


# plt.show()

# def calculateStats():
#     global standardAvvikelseX, standardAvvikelseY;
#     print counter
#     standardAvvikelseX = math.sqrt(standardAvvikelseX/counter)
#     standardAvvikelseY = math.sqrt(standardAvvikelseY /counter)
#     print "Avg X: ", xAvg, ", Avg Y: ", yAvg, ", xMax = ", xMax, ", yMax = ", yMax, " xMin = ", xMin , " yMin = ", yMin
#     print "StandardAvikelseX: ", standardAvvikelseX, ", StandardAvikelseY: ", standardAvvikelseY
#
# print calculateStats()
# # test = "A1,RSSI:-80,MSG:P1"
#
#
#
#
#
#
#
#
# # particle_filter()
