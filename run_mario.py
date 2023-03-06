from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import keyboard
import time 

env = gym_super_mario_bros.make('SuperMarioBros2-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# actions for very simple movement
# SIMPLE_MOVEMENT = [
#     ['NOOP'],
#     ['right'],
#     ['right', 'A'],
#     ['right', 'B'],
#     ['right', 'A', 'B'],
#     ['A'],
#     ['left'],
# ]

done = True
for step in range(5000):
    if done:
        state = env.reset()
    if keyboard.is_pressed('up') and keyboard.is_pressed('right') and keyboard.is_pressed('space'):
        act = 4
    if keyboard.is_pressed('up') and keyboard.is_pressed('right'):
        act = 2
    elif keyboard.is_pressed('space') and keyboard.is_pressed('right'):
        act = 3
    elif keyboard.is_pressed('up'):
        act = 5
    elif keyboard.is_pressed('right'):
        act = 1
    elif keyboard.is_pressed('left'):
        act = 6
    else:
        act = 0
    state, reward, done, info = env.step(act)
    print(reward)
    env.render()

    time.sleep(1/60)


env.close()