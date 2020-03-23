import numpy as np
import math
import torch

'''Import mongodB'''

def compute_SINR(RSRP, numDrones, N0, BW):
    N = BW*N0
    RSRP_lin = np.power(10,.1*RSRP)
    interference = np.repeat(RSRP_lin[:,:,np.newaxis], numDrones, axis=2)
    for i in range (numDrones):
        interference[:,i,i]= 0
    interference = np.sum(interference,axis=1)
    SINR = RSRP - 10*np.log10(N+interference)
    return SINR

def distance(x1,y1,z1,x2,y2,z2):
    out = np.sqrt(np.square(x2-x1) + np.square(y2 - y1)+ np.square(z2 - z1))
    return out

def pathLoss(fc, dronePos, userPos, dAngle):
    numDrones = np.size(dronePos,0)
    numUsers = np.size(userPos,0)
    c = 3e8
    userPosX = np.reshape(np.repeat(userPos[:,0],numDrones,axis=0),(len(userPos),numDrones))
    userPosY = np.reshape(np.repeat(userPos[:, 1], numDrones, axis=0), (len(userPos), numDrones))
    userPosZ = np.reshape(np.repeat(userPos[:, 2], numDrones, axis=0), (len(userPos), numDrones))
    dronePosX = np.reshape(np.repeat(dronePos[:,0],numUsers,axis=0), (numDrones,len(userPos))).T
    dronePosY = np.reshape(np.repeat(dronePos[:, 1], numUsers, axis=0), (numDrones,len(userPos))).T
    dronePosZ = np.reshape(np.repeat(dronePos[:, 2], numUsers, axis=0), (numDrones,len(userPos))).T

    d = distance(userPosX,userPosY,userPosZ,dronePosX,dronePosY,dronePosZ)

    PL = 20*np.log10(4*np.pi*fc*d/c)

    #computes if user is inside the angle of the drone antenna
    droneRad = distance(userPosX,userPosY,0,dronePosX,dronePosY,0)
    angleThresh = dronePos[:,2]*np.tan(np.deg2rad(dAngle/2))

    PL[droneRad > angleThresh] = np.Inf

    return PL

def alloc_users(userPos, dronePos, fc, dAngle, N0, BW, Pt, connectThresh):
    numDrones = np.size(dronePos, 0)
    numUsers = np.size(userPos, 0)

    allocVec = np.zeros((numUsers,1))

    PL = pathLoss(fc,dronePos,userPos,dAngle)

    RSRP = Pt-PL

    SINR = compute_SINR(RSRP,numDrones,N0,BW)

    for contDrone in range(0,numDrones):
        allocVec[SINR[:,contDrone] > connectThresh] = contDrone+1

    return allocVec, SINR

def main():
    numUsers = 10
    numDrones = 2
    dronePos = np.array([[100, 100, 500], [500, 500, 300]])
    userPos = np.zeros((numUsers,3))
    userPos[:,0:2] = np.random.randint(100, 500,[numUsers,2])
    userPos[:,2] = 1.5
    fc = 2.4e9
    dAngle = 60
    Pt = 0
    BW = 200e3
    N0 = 10**(-20.4)
    connectThresh = 40

    allocVec, SINR = alloc_users(userPos,dronePos,fc,dAngle,N0,BW,Pt,connectThresh)
    print(allocVec, '\n', SINR)


if __name__ == "__main__":
    main()

'''
def path_loss(fc, drone_pos, user_pos, directivity_angle):
    numDrones = np.size(drone_pos,1)
    numUsers = np.size(user_pos,1)
    c = 3e8
    # distance in m
    d = distance(np.tile((user_pos[:,1],1,numDrones),
                         np.tile((user_pos[:,2],1,numDrones),
                                 np.tile((user_pos[:,3],1,numDrones),
                                         np.transpose(np.tile((drone_pos[:,1]),numUsers,1),
                                                      np.transpose(np.tile((drone_pos[:,2]),numUsers,1),
                                                                   np.transpose(np.tile((drone_pos[:,3]),numUsers,1))))))))
    # free space path loss dB
    pl = 20*math.log(4*math.pi*fc*d/c,10)
    #Computes if a user is inside the angle of drone antenna.
    radius = distance(np.tile(user_pos(:,1),1,numDrones),np.tile(user_pos(:,2),1,numDrones),0,np.transpose(np.tile(drone_pos(:,1)),numUsers,1),np.transpose(np.tile(drone_pos(:,2)),numUsers,1),0)
    angleThresh = drone_pos(:,3)'*tan(directivity_angle/2)
    pl(radius>angleThresh) = inf
    return pl
'''