import gym

env = gym.make("Taxi-v3", new_step_api=True, render_mode='rgb_array')
env.reset()
print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

state = env.encode(3, 1, 2, 0)  # (taxi row, taxi column, passenger index, destination index)
print("State:", state)

env.s = state

import random
from IPython.display import clear_output

import numpy as np

q_table = np.zeros([env.observation_space.n, env.action_space.n])

# hyperparameter
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# for plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 100):
    state = env.reset()

    epochs, penalties, reward = 0, 0, 0
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:  # 这一步叫做epsilon策略
            # explore action space
            action = env.action_space.sample()
        else:
            # exploit learned values
            action = np.argmax(q_table[state])

        #         next_state, reward, done, info = env.step(action)
        next_state, reward, done, truncated, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

        if i % 100 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")

print("Training finished.\n")
# load cifar10 data from
import torchvision, torch
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(0.5,0.5,0.5), (0.5,0.5,0.5)])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  # 查dataset和dataloader的作用
                                       download=TRUE,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batchsize=4,
                                         shuffle=TRUE, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=TRUE,transform=transform)

def docstring_func_google_typ(parm_a, parm_b, param_c):
    """

    Args:
        parm_a (int):
        parm_b (nn.Module):
        param_c (bool):

    Returns:
        int:
    """

