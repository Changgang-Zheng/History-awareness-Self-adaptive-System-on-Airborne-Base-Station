import numpy as np
import math
import torch
import pymongo
import argparse
import random
import models
from models import SARSA
import pandas as pd
from pymongo import MongoClient
from pandas import DataFrame,Series
import matplotlib.pyplot as plt, time
from matplotlib.patches import Circle
from datetime import datetime
import pickle
import copy

parser = argparse.ArgumentParser(description='Reinforce Learning')
parser.add_argument('--random_seed', default=19, type=int, help='The specific seed to generate the random numbers')
parser.add_argument('--numDrones', default=2, type=int, help='The number of Drones(UAV)')
parser.add_argument('--numUsers', default=1050, type=int, help='The number of Users')
parser.add_argument('--length', default=100, type=int, help='The length of the area(meter)')
parser.add_argument('--width', default=100, type=int, help='The width of the area(meter)')
parser.add_argument('--resolution', default=10, type=int, help='The Resolution (meter)')
parser.add_argument('--episode', default=10000, type=int, help='The number turns it plays')
parser.add_argument('--step', default=20000, type=int, help='The number of steps for any turn of runs')
parser.add_argument('--action_space', default=['east','west','south','north','stay'], type=list, help='The avaliable states')
parser.add_argument('--EPSILON', default=0.9, type=float, help='The greedy policy')
parser.add_argument('--ALPHA', default=0.1, type=float, help='The learning rate')
parser.add_argument('--LAMBDA', default=0.9, type=float, help='The discount factor')

parser.add_argument('--connectThresh', default=40, type=int, help='Threshold')
parser.add_argument('--dAngle', default=60, type=int, help='The directivity angle')
parser.add_argument('--fc', default=2.4e9, type=int, help='The carrier frequency')
parser.add_argument('--Pt', default=0, type=int, help='The drone transmit power in Watts')
parser.add_argument('--BW', default=200e3, type=int, help='The bandwidth')
parser.add_argument('--N0', default=10**(-20.4), type=float, help='The N0')
parser.add_argument('--SIGMA', default=20, type=int, help='The SIGMA')

parser.add_argument('--database_name', default='SARSA_Data_Base', type=str, help='The name of database')
parser.add_argument('--collection_name', default='Q_table_collection', type=str, help='The name of the collection')
parser.add_argument('--host', default='localhost', type=str, help='The host type')
parser.add_argument('--mongodb_port', default=27017, type=int, help='The port of database')


args = parser.parse_args()
sarsa = SARSA(args)

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

def environment_setup():
    np.random.seed(args.random_seed)
    u = np.random.randint(300,700)
    # dronePos = np.zeros((args.numDrones,3))
    # dronePos[:,0:2] = np.random.randint(0, int(args.length/args.resolution),[args.numDrones,2])*10+5
    # dronePos[:,2] = 30
    dronePos = np.array([[5, 5, 30], [95, 95, 30]])

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
    #save_initial_settling(userPos,dronePos)
    Q_table = {}
    for i in range(args.numDrones):
        Q_table[i] = sarsa.build_Q_table()
    return dronePos, userPos, Q_table



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

def save_Q_table(table, SINR, initial_real_reward, action, dronePos, episode, step, drone, name , collection_name, host='localhost', port=27017):
    myclient = pymongo.MongoClient(host='localhost', port=27017)
    mydb = myclient[name]
    dblist = myclient.list_database_names()
    data = {}
    data['episode'] = episode
    data['step'] = step
    data['drone number'] = drone
    drone_dict = data ['qtable'] = {}
    for i in table[int(drone)].index:
        drone_dict['position: ' + i] = {}
        for j in table[int(drone)].columns:
            drone_dict['position: ' + i][j] = table[int(drone)].loc[i, j]
    drone_dict['SINR'] = generate_dict_from_array( SINR, 'user')
    drone_dict['state'] = generate_dict_from_array(dronePos, 'drone')
    drone_dict['action'] = action
    drone_dict['reward'] = initial_real_reward
    collection = mydb[collection_name]
    result = collection.insert(data)

def theoretical_performance(userPos, num):
    max = 0
    average = 0
    for i in range (num):
        np.random.seed(i)
        drone_Pos = np.zeros((args.numDrones, 3))
        drone_Pos[:, 0:2] = np.random.randint(0, int(args.length / args.resolution), [args.numDrones, 2]) * 10 + 5
        drone_Pos[:, 2] = 30
        _, _, initial_real_reward = models.alloc_users(userPos, drone_Pos, args.fc, args.dAngle, args.N0, args.BW, args.Pt,args.connectThresh)
        average += initial_real_reward['total']
        if max<= initial_real_reward['total']:
            max = initial_real_reward['total']
    average /= num
    print('theoretical maximum performance', max, 'theoretical average performance', average)
    return max, average

def main(args):
    #refreash_dataset()

    dronePos, userPos, Q_table = environment_setup()
    #max, average = theoretical_performance(userPos, 100000)


    count = []
    for i in range(args.episode):
        dronePos = np.array([[5, 5, 30], [95, 95, 30]])#----------------------------------------------------------------------------------------------------------------------------------------------------potential:0
        total = 0
        counter = 0
        reward_table = np.zeros((args.numDrones, args.step))
        for j in range(args.step):
            initial_state = dronePos
            initial_table_reward = 0
            second_real_reward = 0
            second_table_reward = 0
            rewards = 0
            for k in range(args.numDrones):
                initial_table_reward, initial_action, Q_table[k] = sarsa.choose_action(dronePos[k][:2], Q_table[k])
                dronePos[k][:2] = sarsa.take_action(dronePos[k][:2], initial_action)
                _, _, initial_real_reward = models.alloc_users(userPos,dronePos,args.fc,args.dAngle,args.N0,args.BW,args.Pt,args.connectThresh)
                second_state = dronePos
                second_table_reward , action, Q_table[k] = sarsa.choose_action(dronePos[k][:2], Q_table[k])
                allocVec, SINR, second_real_reward = models.alloc_users(userPos,dronePos,args.fc,args.dAngle,args.N0,args.BW,args.Pt,args.connectThresh)
                dronePos = second_state #----------------------------------------------------------------------------------------------------------------------------------------------------potential:1
                Q_table[k] = sarsa.update_Q_table(Q_table[k], initial_state[k][:2], initial_action, initial_table_reward, second_state[k][:2], second_table_reward, second_real_reward['total'])
                rewards = initial_real_reward['total']
                save_Q_table(Q_table, SINR, initial_real_reward, action, dronePos, i,j,k, args.database_name, args.collection_name)
            counter += 1
            total += initial_real_reward['total']
            if j%200 ==0:
                print('eisode', i,' with average reward:', total/counter)
            reward_table[k,j] = rewards
        count += [total/counter]
        #np.save('./Log/reward_episod_' + str(i)+ '.npy', count)
        print (count)
        print(Q_table)
        print(dronePos)


if __name__ == "__main__":
    main(args)

