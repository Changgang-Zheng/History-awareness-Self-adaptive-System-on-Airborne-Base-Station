import os
import numpy as np
from PIL import Image, ExifTags
import xlrd
import torch
import cv2
from matplotlib import pyplot as plt
import re
import urllib.request

def load_data(address, args):
    Data = {}
    if args.count ==0:
        np.random.shuffle(address)
    for i in range(args.stride):
        args.count = (args.count +i) % (np.shape(address)[0])
        if args.count+i >=15:
            args.count = 0
        # print(args.count+i)
        env = np.load(address[args.count + i, 0])
        label = np.load(address[args.count + i, 1])
        rew_map = np.load(address[args.count + i, 2])
        env = np.array([env])
        label = np.array([label])
        rew_map = np.array([rew_map])

        env = np.swapaxes(np.swapaxes(env,2,3),1,2)
        # rew_map = np.swapaxes(np.swapaxes(rew_map,2,3),1,2)

        if i == 0:
            env_list = env
            label_list = label
            rew_map_list = rew_map
        else:
            env_list = np.concatenate((env_list, env), axis=0)
            label_list = np.concatenate((label_list, label), axis=0)
            rew_map_list = np.concatenate((rew_map_list, rew_map), axis=0)

    Data['env'] = torch.from_numpy(env_list).to(torch.float32)
    Data['max_posi'] = torch.from_numpy(label_list).to(torch.float32)
    Data['reward_map'] = torch.from_numpy(rew_map_list).to(torch.float32)

    if args.count + args.stride+1 >= np.shape(address)[0]:
        np.random.shuffle(address)
    args.count = (args.count + args.stride+1) % (np.shape(address)[0])
    # print(args.count+args.stride+1,np.shape(address)[0])
    return Data


def dict_to_array(address):
    final_address = []
    for dir in address:
        final_address  += [[address[dir]['env'], address[dir]['max_posi'], address[dir]['reward_map']]]
    return np.array(final_address)

def dataset_description():
    env = load_file('Environment')
    max_posi = load_file('Label')
    reward_map = load_file('Reword Map')
    address = {}
    for i in range (len(env)):
        address[i] = {}
        for j in range(len(env)):
            if int(env[j][env[j].rfind('\\'):][1:-4])==i:
                address[i]['env'] = env[j]
            if int(max_posi[j][max_posi[j].rfind('\\'):][1:-4])==i:
                address[i]['max_posi'] = max_posi[j]
            if int(env[j][env[j].rfind('\\'):][1:-4])==i:
                address[i]['reward_map'] = reward_map[j]
    address = dict_to_array(address)
    return address

def load_file(path):
    file_path = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            file_path += [os.path.join(root, name)]
    return file_path

