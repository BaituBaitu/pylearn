import os
import gym
import torch
import datetime
import numpy as np
import argparse
from rl_utils import ReplayBuffer
from dqn import DQN, MLP

def get_args():
    """ hyperparameters
    """
    curr_time = datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S")  # obtain current time
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--algo_name', default='DQN',
                        type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='CartPole-v1',
                        type=str, help="name of environment")
    parser.add_argument('--train_eps', default=500,
                        type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=20, type=int,
                        help="episodes of testing")
    parser.add_argument('--gamma', default=0.98,
                        type=float, help="discounted factor")
    parser.add_argument('--epsilon', default=0.95,
                        type=float, help="initial value of epsilon")
    parser.add_argument('--epsilon_min', default=0.01,
                        type=float, help="final value of epsilon")
    parser.add_argument('--epsilon_decay', default=500, type=int,
                        help="decay rate of epsilon, the higher value, the slower decay")
    parser.add_argument('--lr', default=0.002,
                        type=float, help="learning rate")
    parser.add_argument('--memory_capacity', default=10000,
                        type=int, help="memory capacity")
    parser.add_argument('--min_memory', default=500, type=int, help="buffer超过这个值之后才开始更新网络")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--target_update_interval', default=10, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--device', default='cpu',
                        type=str, help="cpu or cuda")

    args = parser.parse_args([])
    return args


env_name = 'CartPole-v0'
env = gym.make(env_name, new_step_api=True, render_mode='rgb_array')
state, info = env.reset(seed=1, return_info=True)
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
cfg = get_args()
model = MLP(n_states, n_actions, hidden_dim=cfg.hidden_dim)
agent = DQN(n_states, n_actions, learning_rate=cfg.lr, gamma=cfg.gamma,
            epsilon=cfg.epsilon, epsilon_decay=cfg.epsilon_decay, epsilon_min=cfg.epsilon_min,
            memory_size=cfg.memory_capacity, target_update_interval=cfg.target_update_interval,
            batch_size=cfg.batch_size, device=cfg.device, q_net=model)
def train(env, agent, cfg):
    print("Training start")
    print(f"Env: {env_name}, Algorithm: {cfg.algo_name}, Device: {cfg.device}")
    rewards = []  # record rewards for all episodes
    steps = []
    for i_ep in range(cfg.train_eps):
        ep_reward = 0
        ep_step=0
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.memory.add(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            if agent.memory.size>cfg.min_memory:
                agent.update()
                ep_step += 1
        rewards.append(ep_reward)
        print(f"epsode: {i_ep}, reward: {ep_reward}, ep_step: {ep_step}")
    return rewards

rewards = train(env, agent, cfg)
