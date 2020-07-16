import numpy as np
import pandas as pd

class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False

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
    allocVec = {}
    reward = {}
    for contDrone in range(0, numDrones):
        allocVec[str(contDrone)] = np.zeros((numUsers,1))
    allocVec['total'] = np.zeros((numUsers, 1))

    PL = pathLoss(fc,dronePos,userPos,dAngle)
    RSRP = Pt-PL
    SINR = compute_SINR(RSRP,numDrones,N0,BW)
    for contDrone in range(0,numDrones):
        allocVec[str(contDrone)][SINR[:,contDrone] > connectThresh] = contDrone+1
        allocVec['total'][SINR[:, contDrone] > connectThresh] = contDrone + 1
    for dict in allocVec:
        reward[dict] = 0
        for i in allocVec[dict]:
            if i > 0:
                reward[dict] += 1

    return allocVec, SINR, reward

class Q_Learning:
    def __init__(self, args):
        super(Q_Learning, self).__init__()
        self.args = args

    def build_Q_table(self):
        state_space = []
        for i in range(1, int(self.args.length/self.args.resolution+2)):
            for j in range(1, int(self.args.width/self.args.resolution+2)):
                state_space += [str((int(i*self.args.resolution-self.args.resolution/2),int(j*self.args.resolution-self.args.resolution/2)))]
        Q_table = pd.DataFrame(
            np.zeros((int(self.args.length/self.args.resolution+1)*int(self.args.width/self.args.resolution+1),len(self.args.action_space))),
            columns = self.args.action_space,
            index = state_space,
        )
        return Q_table

    def check_state_exist(self, state, Q_table):
        if str(tuple(map(int,state))) not in Q_table.index:
            # append new state to q table
            Q_table = Q_table.append(
                pd.Series(
                    [0]*len(self.args.action_space),
                    index = Q_table.columns,
                    name = str(tuple(map(int,state))),
                    )
            )
        return Q_table

    def choose_action(self, state, Q_table):
        Q_table = self.check_state_exist(state, Q_table)
        # action selection
        if np.random.rand() < self.args.EPSILON:
            # choose best action
            state_action = Q_table.loc[str(tuple(map(int,state))), :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.args.action_space)
        reward = Q_table.loc[str(tuple(map(int,state))), action]
        return reward, action, Q_table

    def choose_max_action(self, state, Q_table):
        Q_table = self.check_state_exist(state, Q_table)
        # action selection
        # choose best action
        state_action = Q_table.loc[str(tuple(map(int,state))), :]
        # some actions may have the same value, randomly choose on in these actions
        action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        reward = Q_table.loc[str(tuple(map(int,state))), action]
        return reward, action, Q_table

    def take_action(self, state, action):
        for case in switch(action):
            if case('east'):
                if state[0] + self.args.resolution < self.args.length:
                    state[0] += self.args.resolution
                break
            if case('west'):
                if state[0] - self.args.resolution > 0:
                    state[0] -= self.args.resolution
                break
            if case('south'):
                if state[1] - self.args.resolution > 0:
                    state[1] -= self.args.resolution
                break
            if case('north'):
                if state[1] + self.args.resolution < self.args.width:
                    state[1] += self.args.resolution
                break
            if case('stay'):
                state = state
                break
            if case(): # default, could also just omit condition or 'if True'
                print ("State error, which are found in the switch condition !")
                # No need to break here, it'll stop anyway
        return state

    def update_Q_table(self, Q_table, initial_state,initial_action, initial_table_reward, second_state, second_table_reward, second_real_reward):
        Q_table.loc[str(tuple(map(int,initial_state))), initial_action] =  initial_table_reward+self.args.ALPHA*(second_real_reward + self.args.LAMBDA*second_table_reward - initial_table_reward)
        return Q_table











