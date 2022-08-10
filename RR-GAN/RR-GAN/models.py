import numpy as np
import pandas as pd
import copy
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

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
        allocVec[str(contDrone)] = np.zeros((numUsers,numDrones))
    allocVec['total'] = np.zeros((numUsers,numDrones))

    PL = pathLoss(fc,dronePos,userPos,dAngle)
    RSRP = Pt-PL
    SINR = compute_SINR(RSRP,numDrones,N0,BW)
    for contDrone in range(0,numDrones):
        allocVec[str(contDrone)][SINR[:,contDrone] > connectThresh, contDrone] = contDrone+1
        allocVec['total'][SINR[:, contDrone] > connectThresh, contDrone] = contDrone + 1

    for dict in allocVec:
        reward[dict] = 0
        for i in range(allocVec[dict].shape[0]):
            for j in range(allocVec[dict].shape[1]):
                if allocVec[dict][i, j] > 0:
                    reward[dict] += 1
                    break

    return allocVec, SINR, reward


# Deep Q Network off-policy
class Deep_Q_Network:
    def __init__(self, args):
        super(Deep_Q_Network, self).__init__()
        self.args = args

    def observe(self, User_pos):
        state = np.zeros((self.args.length, self.args.width, 1))
        for j in range(self.args.numDrones):
            for i in range(np.shape(User_pos)[0]):
                state[int(User_pos[i, 0]), int(User_pos[i, 1]), 0] = 100
        return state

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

    def pred_loss(self, reward, Q_next, Q_eval, action):
        Q_target = Q_eval.clone()
        #Q_target[0, action] = reward[0] + self.args.LAMBDA * torch.max(Q_next)
        Q_target[0, action] = reward[0] + 0.00001
        predict_loss = nn.MSELoss(reduction='mean')
        loss = predict_loss(Q_target, Q_eval)
        return loss


class net(nn.Module):
    def __init__(self,args):
        super(net, self).__init__()
        self.args = args
        # Encoder layers
        self.enc_conv1_1 = nn.Conv2d(3*self.args.sequence_len , 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_bn1_1 = nn.BatchNorm2d(64)
        self.enc_drop1 = nn.Dropout(p=self.args.drop_rate)
        self.enc_conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_bn1_2 = nn.BatchNorm2d(64)
        self.enc_max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_bn2_1 = nn.BatchNorm2d(128)
        self.enc_drop2 =  nn.Dropout(p=self.args.drop_rate)
        self.enc_conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_bn2_2 = nn.BatchNorm2d(128)
        self.enc_max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_bn3_1 = nn.BatchNorm2d(256)
        self.enc_drop3 = nn.Dropout(p=self.args.drop_rate)
        self.enc_conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_bn3_2 = nn.BatchNorm2d(256)
        self.enc_max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_bn4_1 = nn.BatchNorm2d(512)
        self.enc_drop4 = nn.Dropout(p=self.args.drop_rate)
        self.enc_conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_bn4_2 = nn.BatchNorm2d(512)
        self.enc_avg_pool4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512*6*6, 256)
        self.fc_drop1 = nn.Dropout(p=self.args.drop_rate)
        self.fc2 = nn.Linear(256, len(self.args.action_space))

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.enc_bn1_1(self.enc_conv1_1(x)))
        x = self.enc_max_pool1(self.relu(self.enc_bn1_2(self.enc_conv1_2(x))))
        x = self.relu(self.enc_bn2_1(self.enc_conv2_1(x)))
        x = self.enc_max_pool2(self.relu(self.enc_bn2_2(self.enc_conv2_2(x))))
        x = self.relu(self.enc_bn3_1(self.enc_conv3_1(x)))
        x = self.enc_max_pool3(self.relu(self.enc_bn3_2(self.enc_conv3_2(x))))
        x = self.relu(self.enc_bn4_1(self.enc_conv4_1(x)))
        x = self.enc_avg_pool4(self.relu(self.enc_bn4_2(self.enc_conv4_2(x))))
        x = self.relu(x.reshape(1,512*6*6))
        #x = self.sigmoid(x)
        x = self.fc_drop1(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return 500*self.sigmoid(x)


class SARSA:
    def __init__(self, args):
        super(SARSA, self).__init__()
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
                if state[0] + self.args.resolution < self.args.width:
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











