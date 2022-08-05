# load the environment
import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make("Taxi-v3",new_step_api=True,render_mode='rgb_array').env
# api更新参见 https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
env.reset()
env_screen = np.array(env.render())[0]
print(env_screen.shape)
print("Action Space: {}".format(env.action_space))
print("State Space: {}".format(env.observation_space))




