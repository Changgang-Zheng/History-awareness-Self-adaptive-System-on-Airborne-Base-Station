import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import config as cf
import math
import pymongo
import argparse
import random
import models
import os
from models import SARSA
from models import Deep_Q_Network, net
import pandas as pd
from pymongo import MongoClient
from pandas import DataFrame,Series
import matplotlib.pyplot as plt, time
from matplotlib.patches import Circle
import pickle
import copy
from random import sample

parser = argparse.ArgumentParser(description='Reinforce Learning')
#=======================================================================================================================
# Environment Parameters
parser.add_argument('--random_seed', default=19, type=int, help='The specific seed to generate the random numbers')
parser.add_argument('--numDrones', default=2, type=int, help='The number of Drones(UAV)')
parser.add_argument('--numUsers', default=1050, type=int, help='The number of Users')
parser.add_argument('--length', default=100, type=int, help='The length of the area(meter)')
parser.add_argument('--width', default=100, type=int, help='The width of the area(meter)')
parser.add_argument('--resolution', default=10, type=int, help='The Resolution (meter) for drones')
parser.add_argument('--episode', default=100, type=int, help='The number turns it plays')
parser.add_argument('--step', default=2000, type=int, help='The number of steps for any turn of runs')
parser.add_argument('--round', default=200, type=int, help='The number of rounds per training')
parser.add_argument('--interval', default=200, type=int, help='The interval between each chunk of training rounds')
parser.add_argument('--action_space', default=['east','west','south','north','stay'], type=list, help='The avaliable states')
parser.add_argument('--EPSILON', default=0.8, type=float, help='The greedy policy')
parser.add_argument('--ALPHA', default=0.01, type=float, help='The learning rate')
parser.add_argument('--LAMBDA', default=0.1, type=float, help='The discount factor')
parser.add_argument('--store_step', default=100, type=int, help='number of steps per storation, store the data from target network')
#=======================================================================================================================
# DQN Parameters
parser.add_argument('--lr', default=0.005, type=float, help='The learning rate for CNN')
parser.add_argument('--drop_rate', default=0.5, type=float, help='The drop out rate for CNN')
parser.add_argument('--iteration', default=1, type=int, help='The number of data per train')
parser.add_argument('--sequence_len', default=1, type=int, help='The number of observations in a sequence')
#=======================================================================================================================
# Signal Attenuation Parameters
parser.add_argument('--connectThresh', default=40, type=int, help='Threshold')
parser.add_argument('--dAngle', default=60, type=int, help='The directivity angle')
parser.add_argument('--fc', default=2.4e9, type=int, help='The carrier frequency')
parser.add_argument('--Pt', default=0, type=int, help='The drone transmit power in Watts')
parser.add_argument('--BW', default=200e3, type=int, help='The bandwidth')
parser.add_argument('--N0', default=10**(-20.4), type=float, help='The N0')
parser.add_argument('--SIGMA', default=20, type=int, help='The SIGMA')
#=======================================================================================================================
# Database Parameters
parser.add_argument('--database_name', default='DQN_Data_Base2', type=str, help='The name of database')
parser.add_argument('--collection_name', default='Q_table_collection', type=str, help='The name of the collection')
parser.add_argument('--host', default='localhost', type=str, help='The host type')
parser.add_argument('--mongodb_port', default=27017, type=int, help='The port of database')


args = parser.parse_args()
sarsa = SARSA(args)
DQN = Deep_Q_Network(args)


def generate_pre_Q_dict_from_array(array):
    dict = {}
    for i in range(np.shape(array)[0]):
        data = '( '
        for j in range(np.shape(array)[1]):
            if j != 0:
                data += ', '
            data += str(array[i,j])
        data += ' )'
        dict [str(args.action_space[i])] = data
    return copy.deepcopy(dict)

def generate_dict_from_array(array, name):
    dict = {}
    for i in range(np.shape(array)[0]):
        data = '( '
        for j in range(np.shape(array)[1]):
            if j != 0:
                data += ', '
            data += str(array[i,j])
        data += ' )'
        dict [name + ' ' + str(i)] = data
    return copy.deepcopy(dict)

def environment_setup():
    np.random.seed(args.random_seed)
    u = np.random.randint(300,700)
    # dronePos = np.zeros((args.numDrones,3))
    # dronePos[:,0:2] = np.random.randint(0, int(args.length/args.resolution),[args.numDrones,2])*10+5
    # dronePos[:,2] = 30
    dronePos = np.array([[5, 5, 30], [99, 99, 30]])
    #dronePos = np.array([[0, 0, 30], [99, 99, 30], [0, 99, 30], [99, 0, 30]]
    #dronePos = np.array([[0, 0, 30], [99, 99, 30], [0, 99, 30], [99, 0, 30], [0, 49, 30], [49, 0, 30], [99, 49, 30], [49, 99, 30]])


    # userPos = np.zeros((args.numUsers,3))
    # userPos[:,0:2] =np.floor((np.random.randn(args.numUsers,2)*args.SIGMA*5 + u)%args.length)
    # userPos[:,2] = 1.5
    resolution = 10
    length = 100

    cluster = {}
    number = {}
    colour = {}
    SIGMA = {}
    u = {}
    label = {}
    # font = {'family': 'Palatino',}
    font = {}
    u['cluster1'] = [np.random.randint(20, 80), np.random.randint(20, 80)]
    SIGMA['cluster1'] = 7
    number['cluster1'] = 250
    colour['cluster1'] = '#9400D3'
    cluster['cluster1'] = np.zeros((number['cluster1'], 3))
    cluster['cluster1'][:, 0:1] = np.floor(
        (np.random.randn(number['cluster1'], 1) * SIGMA['cluster1'] + u['cluster1'][0]) % length)
    cluster['cluster1'][:, 1:2] = np.floor(
        (np.random.randn(number['cluster1'], 1) * SIGMA['cluster1'] + u['cluster1'][1]) % length)
    label['cluster1'] = 'MS cluster 1'

    u['cluster2'] = [np.random.randint(30, 80), np.random.randint(20, 70)]
    SIGMA['cluster2'] = 10
    number['cluster2'] = 300
    colour['cluster2'] = '#FF8C00'
    cluster['cluster2'] = np.zeros((number['cluster2'], 3))
    cluster['cluster2'][:, 0:1] = np.floor(
        (np.random.randn(number['cluster2'], 1) * SIGMA['cluster2'] + u['cluster2'][0]) % length)
    cluster['cluster2'][:, 1:2] = np.floor(
        (np.random.randn(number['cluster2'], 1) * SIGMA['cluster2'] + u['cluster2'][1]) % length)
    label['cluster2'] = 'MS cluster 2'

    u['cluster3'] = [np.random.randint(10, 85), np.random.randint(10, 90)]
    SIGMA['cluster3'] = 6
    number['cluster3'] = 200
    colour['cluster3'] = '#228B22'
    cluster['cluster3'] = np.zeros((number['cluster3'], 3))
    cluster['cluster3'][:, 0:1] = np.floor(
        (np.random.randn(number['cluster3'], 1) * SIGMA['cluster3'] + u['cluster3'][0]) % length)
    cluster['cluster3'][:, 1:2] = np.floor(
        (np.random.randn(number['cluster3'], 1) * SIGMA['cluster3'] + u['cluster3'][1]) % length)
    label['cluster3'] = 'MS cluster 3'

    number['uniform'] = 300
    cluster['uniform'] = np.random.randint(0, 100, size=[number['uniform'], 3])
    colour['uniform'] = '#4169E1'
    label['uniform'] = 'MS uniform'
    for dict in cluster:
        if dict == 'cluster1':
            userPos = cluster[dict]
        else:
            userPos = np.concatenate((userPos, cluster[dict]), axis=0)
    userPos[:, 2] = 1.5
    # save_initial_settling(userPos,dronePos)
    return dronePos, userPos


def refreash_dataset(name = args.database_name, collection_name = args.collection_name, host='localhost', port=27017):
    mongo_client = MongoClient(host, port) # 创建 MongoClient 对象，（string格式，int格式）
    mongo_db = mongo_client[name] # MongoDB 中可存在多个数据库，根据数据库名称获取数据库对象（Database）
    #mongo_db.authenticate(mongodb_user, mongodb_passwd) # 登录认证
    for collection in mongo_db.collection_names():
        print ('collection: ', collection, ' have been refreshed')
        db_collection=mongo_db[collection] # 每个数据库包含多个集合，根据集合名称获取集合对象（Collection）
        # drop = db_collection.drop() delate
        remove = db_collection.remove()
        #drop = db_collection.drop()


def save_initial_settling(U_p, D_p, name = args.database_name, collection_name ='initial_setting', host='localhost', port=27017):
    myclient = pymongo.MongoClient(host='localhost', port=27017)
    mydb = myclient[name]
    dblist = myclient.list_database_names()
    collection = mydb[collection_name]
    initial_info = {}
    initial_info ['random_seed'] = args.random_seed
    initial_info ['num_drones'] = args.numDrones
    initial_info ['num_users'] = args.numUsers
    initial_info ['user_positions'] = generate_dict_from_array(U_p, 'user')
    initial_info ['drone_positions'] = generate_dict_from_array(D_p, 'drone')
    initial_info ['carrier_frequency'] = args.fc
    initial_info ['transmit_power'] = args.Pt
    initial_info ['sinr_threshold'] = args.connectThresh
    initial_info ['drone_user_capacity'] = 'not consider yet'
    initial_info ['x_min'] = 0
    initial_info ['x_max'] = args.width
    initial_info ['y_min'] = 0
    initial_info ['y_max'] = args.length
    initial_info ['possible_actions'] = [[1,0],[-1,0],[0,1],[0,-1],[0,0]]
    initial_info ['learning_rate'] = args.ALPHA
    initial_info ['total_episodes'] = args.episode
    initial_info ['iterations_per_episode'] = args.step
    initial_info ['discount_factor'] = args.LAMBDA
    initial_info ['episodes'] = 'total if possible'
    result = collection.insert(initial_info)

def save_Q_table(table, SINR, initial_real_reword, action, dronePos, episode, step, drone, name , collection_name, host='localhost', port=27017):
    myclient = pymongo.MongoClient(host='localhost', port=27017)
    mydb = myclient[name]
    dblist = myclient.list_database_names()
    data = {}
    epoch_dict = data['episode: ' + episode] = {}
    step_dict = epoch_dict['step: ' + step] = {}
    drone_dict = step_dict ['drone number: ' + drone] = {}
    for i in table[int(drone)].index:
        drone_dict['position: ' + i] = {}
        for j in table[int(drone)].columns:
            drone_dict['position: ' + i][j] = table[int(drone)].loc[i, j]
    drone_dict['SINR'] = generate_dict_from_array( SINR, 'user')
    drone_dict['state'] = generate_dict_from_array(dronePos, 'drone')
    drone_dict['action'] = action
    drone_dict['reword'] = initial_real_reword
    collection = mydb[collection_name]
    result = collection.insert(data)
    #print(result)

def save_predicted_Q_table(observation_seq, SINR, predicted_table, action, reward_, dronePos, episode, step, drone, name , collection_name, host='localhost', port=27017):
    myclient = pymongo.MongoClient(host='localhost', port=27017)
    mydb = myclient[name]
    dblist = myclient.list_database_names()
    data = {}
    epoch_dict = data['episode: ' + episode] = {}
    step_dict = epoch_dict['step: ' + step] = {}
    drone_dict = step_dict ['drone number: ' + drone] = {}
    drone_dict['position: (' + str(dronePos[int(drone),0])+', '+str(dronePos[int(drone),1])+')'] = {}
    drone_dict['position: (' + str(dronePos[int(drone),0])+', '+str(dronePos[int(drone),1])+')'] = generate_pre_Q_dict_from_array(predicted_table.T)
    drone_dict['SINR'] = generate_dict_from_array( SINR, 'user')
    drone_dict['state'] = generate_dict_from_array(dronePos, 'drone')
    drone_dict['action'] = action
    drone_dict['reword'] = reward_
    collection = mydb[collection_name]
    result = collection.insert(data)
    #print(result)

def save_data_for_training(Store_transition, count, observation_seq_adjust, action_adjust, reward_, observation_seq_adjust_):
    Store_transition[count] = {}
    # Store_transition[count]['observation_seq'] = np.array([observation_seq_adjust])
    Store_transition[count]['observation_seq'] = np.array([observation_seq_adjust])
    Store_transition[count]['action'] = action_adjust
    Store_transition[count]['reward_'] = np.array([reward_])
    Store_transition[count]['observation_seq_'] = np.array([observation_seq_adjust_])
    if (count+1) % args.store_step == 0 and count != 0:
        np.save('Data\\'+str(count - args.store_step + 1) + '_to_' + str(count) + '.npy', Store_transition)
        Store_transition = {}
    # np.save('Data\\' + str(count - count%args.store_step ) + '_to_' + str(count - count%args.store_step + args.store_step - 1) + '.npy', Store_transition)
    return Store_transition

def grasp_data_for_training(Store_transition, count, numbers = 1):
    selected = sample([i for i in range(count)], numbers)
    for dict in selected:
        if (count - (count)%args.store_step) <= dict :
            Store = Store_transition
        else:
            Store = np.load('Data\\' + str(dict - dict%args.store_step ) + '_to_' + str(dict - dict%args.store_step + args.store_step - 1) + '.npy', allow_pickle = True).item()

        if ('r_' not in dir()) or ('state_' not in dir()):
            state = Store[dict]['observation_seq']
            state_ = Store[dict]['observation_seq_']
            r_ = Store[dict]['reward_']
            action = Store[dict]['action']
        else:
            state = np.concatenate((state, Store[dict]['observation_seq']), axis=0)
            state_ = np.concatenate((state_, Store[dict]['observation_seq_']), axis=0)
            # state = Store[dict]['observation_seq']
            r_ = np.concatenate((r_, Store[dict]['reward_']), axis=0)
            action = Store[dict]['action']
    return  state, r_, action, state_

def plot_double(usr, env):
    plt.figure(12)
    plt.subplot(121)

    plt.xlim((0, 100))  # 设置坐标轴范围
    plt.ylim((0, 100))
    plt.xticks(np.arange(0, 100, 10))
    plt.yticks(np.arange(0, 100, 10))
    plt.scatter(usr[:, 0], usr[:, 1], alpha=0.3, edgecolors='none')

    plt.subplot(122)
    # 数据数目
    n = 100
    # 定义x, y
    x = np.linspace(0, 100, n)
    y = np.linspace(0, 100, n)

    # 生成网格数据
    X, Y = np.meshgrid(x, y)

    plt.xlim((0, 100))  # 设置坐标轴范围
    plt.ylim((0, 100))
    plt.xticks(np.arange(0, 100, 10))
    plt.yticks(np.arange(0, 100, 10))
    # 填充等高线的颜色, 8是等高线分为几部分
    plt.contourf(X, Y, np.around(env, -1)[:,:,0] / 40, 6, alpha=0.85, cmap=plt.cm.Blues)
    # 绘制等高线
    plt.grid(ls='--', c='#D3D3D3')
    C = plt.contour(X, Y, np.around(env, -1)[:,:,0] / 40, 6, colors='black', linewidth=0.01)
    # 绘制等高线数据

    plt.show()
    plt.close()


def generator_for_one_drone(args):


    # =============================================== start up =========================================================

    for data in range(100000000):
        dronePos, userPos = environment_setup()



        environment = DQN.observe(userPos)
        label = np.zeros(np.shape(environment))

        max_reward = 0
        if data%1000 == 0 and data!=0:
            if not continues:
                print('stop at'+str(data)+'th data')
                break
            continues = False
        for i in range(np.shape(environment)[0]):
            for j in range(np.shape(environment)[1]):
                dronePos[:, :2] = np.array([i, j])
                _, _, reward = models.alloc_users(userPos, dronePos, args.fc, args.dAngle, args.N0, args.BW, args.Pt,args.connectThresh)
                label[i,j]= reward['total']
                if reward['total'] > max_reward:
                    max_reward = reward['total']
                    location = [i,j]
                    continues = True


        #plot_double(userPos,label)
        np.save('Environment/one/' + str(data) + 'standard.npy', environment)
        np.save('Reword Map/one/' + str(data) + 'standard.npy', label)
        np.save('Label/one/' + str(data) + 'standard.npy', location)

        print(str(data)+'th data is generated and saved')

def generator_for_two_drone(args):
    no = 0
    total = 0

    # =============================================== start up =========================================================

    for data in range(1000):
        dronePos, userPos = environment_setup()

        environment = DQN.observe(userPos)
        label = np.zeros((args.numDrones,np.shape(environment)[0],np.shape(environment)[1]))


        continues = True
        max_reward = 0

        for i in range(10000000):
            if i % 100000 == 0 and i != 0:
                if not continues:
                    # print('stop at' + str(i) + 'th data')
                    break
                continues = False
            np.random.seed(i)
            dronePos[:, :2] = np.array([[random.randint(1, 19) * 5, random.randint(1, 19) * 5],
                                        [random.randint(1, 19) * 5, random.randint(1, 19) * 5]])
            _, _, reward = models.alloc_users(userPos, dronePos, args.fc, args.dAngle, args.N0, args.BW, args.Pt,args.connectThresh)
            if reward['total'] > max_reward:
                max_reward = reward['total']
                max_dronePos = copy.deepcopy(dronePos)
                continues = True
        # print(max_reward)

        # for k in range(args.numDrones):
        #     dronePos = copy.deepcopy(max_dronePos)
        #     for i in range(np.shape(environment)[0]):
        #         for j in range(np.shape(environment)[1]):
        #             dronePos[k,:]=[i,j,30]
        #             _, _, reward = models.alloc_users(userPos, dronePos, args.fc, args.dAngle, args.N0, args.BW, args.Pt,args.connectThresh)
        #             label[k,i,j]= reward['total']
        #
        #
        #
        #
        #
        # #plot_double(userPos,label)
        # np.save('Environment/two/' + str(data) + 'standard.npy', environment)
        # np.save('Reword Map/two/' + str(data) + 'standard.npy', label)
        #
        #
        # print(str(data)+'th data is generated and saved',max_reward)

        no += 1
        total += max_reward
        print(str(data) + 'th averga is: ' + str(total / no))



def generator_for_four_drone(args):


    # =============================================== start up =========================================================

    for data in range(10000):
        dronePos, userPos = environment_setup()

        environment = DQN.observe(userPos)
        label = np.zeros((args.numDrones,np.shape(environment)[0],np.shape(environment)[1]))



        max_reward = 0
        for i in range(10000):
            np.random.seed(i)
            dronePos[:, :2] = np.array([[random.randint(1, 19) * 5, random.randint(1, 19) * 5],
                                        [random.randint(1, 19) * 5, random.randint(1, 19) * 5],
                                        [random.randint(1, 19) * 5, random.randint(1, 19) * 5],
                                        [random.randint(1, 19) * 5, random.randint(1, 19) * 5]
                                        ])
            _, _, reward = models.alloc_users(userPos, dronePos, args.fc, args.dAngle, args.N0, args.BW, args.Pt,args.connectThresh)
            if reward['total'] >= max_reward:
                max_reward = reward['total']
                max_dronePos = copy.deepcopy(dronePos)

        print(max_reward)

        for k in range(args.numDrones):
            dronePos = copy.deepcopy(max_dronePos)
            for i in range(np.shape(environment)[0]):
                for j in range(np.shape(environment)[1]):
                    dronePos[k,:]=[i,j,30]
                    _, _, reward = models.alloc_users(userPos, dronePos, args.fc, args.dAngle, args.N0, args.BW, args.Pt,args.connectThresh)
                    label[k,i,j]= reward['total']




        #plot_double(userPos,label)
        np.save('Environment/four/' + str(data) + 'standard.npy', environment)
        np.save('Reword Map/four/' + str(data) + 'standard.npy', label)


        print(str(data)+'th data is generated and saved',max_reward)

def generator_for_four_drone(args):


    # =============================================== start up =========================================================

    for data in range(10000):
        dronePos, userPos = environment_setup()

        environment = DQN.observe(userPos)
        label = np.zeros((args.numDrones,np.shape(environment)[0],np.shape(environment)[1]))



        max_reward = 0
        for i in range(10000):
            np.random.seed(i)
            dronePos[:, :2] = np.array([[random.randint(1, 19) * 5, random.randint(1, 19) * 5],
                                        [random.randint(1, 19) * 5, random.randint(1, 19) * 5],
                                        [random.randint(1, 19) * 5, random.randint(1, 19) * 5],
                                        [random.randint(1, 19) * 5, random.randint(1, 19) * 5],
                                        [random.randint(1, 19) * 5, random.randint(1, 19) * 5],
                                        [random.randint(1, 19) * 5, random.randint(1, 19) * 5],
                                        [random.randint(1, 19) * 5, random.randint(1, 19) * 5],
                                        [random.randint(1, 19) * 5, random.randint(1, 19) * 5]
                                        ])
            _, _, reward = models.alloc_users(userPos, dronePos, args.fc, args.dAngle, args.N0, args.BW, args.Pt,args.connectThresh)
            if reward['total'] >= max_reward:
                max_reward = reward['total']
                max_dronePos = copy.deepcopy(dronePos)
        print(max_reward)

        for k in range(args.numDrones):
            dronePos = copy.deepcopy(max_dronePos)
            for i in range(np.shape(environment)[0]):
                for j in range(np.shape(environment)[1]):
                    dronePos[k,:]=[i,j,30]
                    _, _, reward = models.alloc_users(userPos, dronePos, args.fc, args.dAngle, args.N0, args.BW, args.Pt,args.connectThresh)
                    label[k,i,j]= reward['total']
                    if max_dronePos[k,:]  == [i, j, 30]:
                        label[k, i, j] += 20
                        print (i,j)




        #plot_double(userPos,label)
        np.save('Environment/eight/' + str(data) + 'standard.npy', environment)
        np.save('Reword Map/eight/' + str(data) + 'standard.npy', label)


        print(str(data)+'th data is generated and saved',max_reward)

if __name__ == "__main__":
    #generator_for_one_drone(args)
    generator_for_two_drone(args)

