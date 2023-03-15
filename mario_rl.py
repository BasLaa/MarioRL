from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
import make_env

import time 


env = gym_super_mario_bros.make('SuperMarioBros2-v0')
env = make_env.make_env(env)
env = JoypadSpace(env, RIGHT_ONLY)

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=4000)
# model.save("ppo_mario")

# del model
# model = PPO.load('ppo_mario', env=env)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(5000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    print(reward)
    vec_env.render()
    time.sleep(1/60)