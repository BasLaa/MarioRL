from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO

import keyboard
import time 


env = gym_super_mario_bros.make('SuperMarioBros2-v0')
# env = JoypadSpace(env, SIMPLE_MOVEMENT)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=4000)
model.save("ppo_mario")

# del model
# model = PPO.load('ppo_mario', env=env)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(5000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    print(reward)
    vec_env.render()
    time.sleep(1/120)