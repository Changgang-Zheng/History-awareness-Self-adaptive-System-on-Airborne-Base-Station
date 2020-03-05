import numpy as np
import random

def dynamic(userPos, epoch, step, args):
    action_space = [-1, 1, 0]
    # if step%20000 == 0:
    #     args.cluster1[0] = args.cluster1[0]+random.sample(action_space, 1)[0]
    #     args.cluster1[1] = args.cluster1[1] + random.sample(action_space, 1)[0]
    # if step%20000 == 0:
    #     args.cluster2[0] = args.cluster2[0]+random.sample(action_space, 1)[0]
    #     args.cluster2[1] = args.cluster2[1] + random.sample(action_space, 1)[0]
    # if step%20000 == 0:
    #     args.cluster3[0] = args.cluster3[0]+random.sample(action_space, 1)[0]
    #     args.cluster3[1] = args.cluster3[1] + random.sample(action_space, 1)[0]

    np.random.seed(args.random_seed)
    u = np.random.randint(300,700)
    # dronePos = np.zeros((args.numDrones,3))
    # dronePos[:,0:2] = np.random.randint(0, int(args.length/args.resolution),[args.numDrones,2])*10+5
    # dronePos[:,2] = 30
    dronePos = np.array([[0, 0, 30], [99, 99, 30]])

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
    u['cluster1'] = args.cluster1
    SIGMA['cluster1'] = 7
    number['cluster1'] = 250
    colour['cluster1'] = '#9400D3'
    cluster['cluster1'] = np.zeros((number['cluster1'], 3))
    cluster['cluster1'][:, 0:1] = np.floor(
        (np.random.randn(number['cluster1'], 1) * SIGMA['cluster1'] + u['cluster1'][0]) % length)
    cluster['cluster1'][:, 1:2] = np.floor(
        (np.random.randn(number['cluster1'], 1) * SIGMA['cluster1'] + u['cluster1'][1]) % length)
    label['cluster1'] = 'MS cluster 1'

    u['cluster2'] = args.cluster2
    SIGMA['cluster2'] = 10
    number['cluster2'] = 300
    colour['cluster2'] = '#FF8C00'
    cluster['cluster2'] = np.zeros((number['cluster2'], 3))
    cluster['cluster2'][:, 0:1] = np.floor(
        (np.random.randn(number['cluster2'], 1) * SIGMA['cluster2'] + u['cluster2'][0]) % length)
    cluster['cluster2'][:, 1:2] = np.floor(
        (np.random.randn(number['cluster2'], 1) * SIGMA['cluster2'] + u['cluster2'][1]) % length)
    label['cluster2'] = 'MS cluster 2'

    u['cluster3'] = args.cluster3
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


    return userPos