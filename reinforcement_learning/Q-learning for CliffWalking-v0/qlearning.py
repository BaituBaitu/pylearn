import numpy as np
import math
import torch
from collections import defaultdict


class QLearning:
    """强化学习算法一般包括sample（训练时采样动作），predict（测试时预测动作），update算法更新，load，save等几个方法"""
    def __init__(self, n_states: int, n_actions: int, **kwargs):
        """
        :param n_states: 状态空间的维度
        :param n_actions: 动作空间的维度
        :param kwargs: 其他参数
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = kwargs.get('gamma', 0.9)
        self.alpha = kwargs.get('alpha', 0.1)  # learning rate
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.99)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        self.sample_count= 0

    def choose_action(self, state: int) -> int:
        """采样动作，根据epsilon-greedy策略选择动作

        :param state: 状态
        :return: 动作
        """
        self.sample_count += 1
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # epsilon是会递减的，这里选择指数递减
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_table[str(state)])

    def predict(self, state: int) -> int:
        """预测动作

        :param state: 状态
        """
        return np.argmax(self.q_table[str(state)])

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """算法更新

        Args:
            state (int):
            action (int):
            reward (float):
            next_state (int):
            done (bool):
        Returns:

        """
        Q_predict = self.q_table[str(state)][action]
        if done:  # 如果next_state是终止状态
            Q_target = reward
        else:
            Q_target = reward + self.gamma * np.max(self.q_table[str(next_state)])
        self.q_table[str(state)][action] += self.alpha * (Q_target - Q_predict)

    def save(self, path, filename):
        import dill
        torch.save(
            obj=self.q_table,
            f=path + filename + '.pkl',
            pickle_module=dill
        )
        print("保存模型成功！")

    def load(self, filename):
        import dill
        self.q_table = torch.load(f=filename, pickle_module=dill)
        print("加载模型成功！")
