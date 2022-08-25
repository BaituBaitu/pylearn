import torch
import copy
import numpy as np
from rl_utils import ReplayBuffer
import torch.nn as nn
import torch.nn.functional as F  # for the activation function


class MLP(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=128):
        """ 初始化q网络，为全连接网络
            n_states: 输入的特征数即环境的状态维度
            n_actions: 输出的动作维度
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)  # 输入层
        self.fc2 = nn.Linear(hidden_dim, n_actions)  # 输出层

    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN:
    """
    经典DQN，处理动作空间为离散数据的情况
    target计算公式为 y=r_i + max_a(Q_tar(s_{i+1}, a))
    参考 https://datawhalechina.github.io/easy-rl/#/chapter6/chapter6
    https://www.heywhale.com/mw/project/611247d8fe727700176c13da
    """

    def __init__(self, n_states: int, n_actions: int, learning_rate: float = 0.001, gamma: float = 0.98,
                 epsilon: float = 0.9, epsilon_decay: float = 0.99, epsilon_min: float = 0.01, memory_size: int = 1e4,
                 target_update_interval: int = 100, batch_size: int = 32, device='cpu', q_net: nn.Module = None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = ReplayBuffer(memory_size)
        self.target_update_interval = target_update_interval
        self.batch_size = batch_size
        self.device = device
        # 对于一般网络的结构，输入为state，输出为包含每一个action的q值的向量，
        # 所以注意q网络的结构，输入维数同state，输出维数为action维数
        self.q_net = q_net.to(self.device)
        self.q_target = copy.deepcopy(q_net).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=self.learning_rate)
        self.count = 0

    def choose_action(self, state):
        """训练时用epsilon-greedy方法选择action"""
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # epsilon是会递减的，这里选择指数递减
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float)
                action = self.q_net(state).argmax().item()
        return action

    def predict(self, state):
        """预测时用，根据状态选择最优action"""
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            action = self.q_net(state).argmax().item()
        return action

    def update(self) -> None:
        """e"""
        if self.memory.size < self.batch_size:  # 当memory中不满足一个批量时，不更新策略
            return
        # sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        state_batch = torch.tensor(state_batch, device=self.device,
                                   dtype=torch.float)  # shape(batchsize,n_states)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)  # shape(batchsize,1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float).unsqueeze(
            1)  # shape(batchsize)
        next_state_batch = torch.tensor(next_state_batch, device=self.device,
                                        dtype=torch.float)  # shape(batchsize,n_states)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device).unsqueeze(1)  # shape(batchsize,1)
        # 计算target
        max_next_q_values = self.q_target(next_state_batch).max(1)[0].unsqueeze(1)  # 下个状态的最大Q值
        q_target_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)  # TD目标
        # 计算当前状态的Q
        q_values = self.q_net(state_batch).gather(dim=1, index=action_batch)  # 计算当前状态(s_t,a)对应的Q(s_t, a)
        # 计算loss
        loss = nn.MSELoss()(q_values, q_target_values)
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.q_net.parameters():  # clip防止梯度爆炸
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.count % self.target_update_interval == 0:
            self.q_target.load_state_dict(self.q_net.state_dict())
        self.count += 1



    def save(self, path):
            torch.save(self.q_target.state_dict(), path + 'checkpoint.pth')

    def load(self, path):
        self.q_target.load_state_dict(torch.load(path + 'checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)


