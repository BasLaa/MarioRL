from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import keyboard
import time
import cv2
import os

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
max = -100
take_screenshot = False
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    if max <= reward and take_screenshot:
        max = reward
        if not os.path.exists('frames'):
            os.makedirs('frames')
        frame = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"frames/frame_{step}.png", frame)
    env.render()

env.close()

#%%
