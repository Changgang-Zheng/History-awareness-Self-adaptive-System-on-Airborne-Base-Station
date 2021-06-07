import tensorflow as tf
import numpy as np
from collections import deque
import random


class DeepQNetwork:
    r = np.array([[-1, -1, -1, -1, 0, -1],
                  [-1, -1, -1, 0, -1, 100.0],
                  [-1, -1, -1, 0, -1, -1],
                  [-1, 0, 0, -1, 0, -1],
                  [0, -1, -1, 1, -1, 100],
                  [-1, 0, -1, -1, 0, 100],
                  ])

    # 执行步数。
    step_index = 0

    # 状态数。
    state_num = 6

    # 动作数。
    action_num = 6

    # 训练之前观察多少步。
    OBSERVE = 1000.

    # 选取的小批量训练样本数。
    BATCH = 20

    # epsilon 的最小值，当 epsilon 小于该值时，将不在随机选择行为。
    FINAL_EPSILON = 0.0001

    # epsilon 的初始值，epsilon 逐渐减小。
    INITIAL_EPSILON = 0.1

    # epsilon 衰减的总步数。
    EXPLORE = 3000000.

    # 探索模式计数。
    epsilon = 0

    # 训练步数统计。
    learn_step_counter = 0

    # 学习率。
    learning_rate = 0.001

    # γ经验折损率。
    gamma = 0.9

    # 记忆上限。
    memory_size = 5000

    # 当前记忆数。
    memory_counter = 0

    # 保存观察到的执行过的行动的存储器，即：曾经经历过的记忆。
    replay_memory_store = deque()

    # 生成一个状态矩阵（6 X 6），每一行代表一个状态。
    state_list = None

    # 生成一个动作矩阵。
    action_list = None

    # q_eval 网络。
    q_eval_input = None
    action_input = None
    q_target = None
    q_eval = None
    predict = None
    loss = None
    train_op = None
    cost_his = None
    reward_action = None

    # tensorflow 会话。
    session = None

    def __init__(self, learning_rate=0.001, gamma=0.9, memory_size=5000):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.memory_size = memory_size

        # 初始化成一个 6 X 6 的状态矩阵。
        self.state_list = np.identity(self.state_num)

        # 初始化成一个 6 X 6 的动作矩阵。
        self.action_list = np.identity(self.action_num)

        # 创建神经网络。
        self.create_network()

        # 初始化 tensorflow 会话。
        self.session = tf.InteractiveSession()

        # 初始化 tensorflow 参数。
        self.session.run(tf.initialize_all_variables())

        # 记录所有 loss 变化。
        self.cost_his = []

    def create_network(self):
        """
        创建神经网络。
        :return:
        """
        pass

    def select_action(self, state_index):
        """
        根据策略选择动作。
        :param state_index: 当前状态。
        :return:
        """
        pass

    def save_store(self, current_state_index, current_action_index, current_reward, next_state_index, done):
        """
        保存记忆。
        :param current_state_index: 当前状态 index。
        :param current_action_index: 动作 index。
        :param current_reward: 奖励。
        :param next_state_index: 下一个状态 index。
        :param done: 是否结束。
        :return:
        """
        pass

    def step(self, state, action):
        """
        执行动作。
        :param state: 当前状态。
        :param action: 执行的动作。
        :return:
        """
        pass

    def experience_replay(self):
        """
        记忆回放。
        :return:
        """
        pass

    def train(self):
        """
        训练。
        :return:
        """
        pass

    def pay(self):
        """
        运行并测试。
        :return:
        """
        pass


if __name__ == "__main__":
    q_network = DeepQNetwork()
    q_network.pay()