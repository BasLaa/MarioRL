import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from stable_baselines3 import PPO
import make_env

import time 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def run_model(pretrained=False, model_name="mario_rl"):
    env = gym_super_mario_bros.make('SuperMarioBros-v0')

    # skip is number of frames that are skipped
    env = make_env.make_env(env, skip=4, move_set=RIGHT_ONLY)

    if pretrained and os.path.isfile(f'models/{model_name}.zip'):
        print("Found existing model...")
        model = PPO.load(f'models/{model_name}', env=env)
    elif pretrained:
        print("Model not found...")
        return
    else:
        print("Training new model...")
        model = PPO("MlpPolicy", env, verbose=1, n_epochs=10, n_steps=3000, batch_size=100)
        model.learn(total_timesteps=4000,)
        model.save(f"models/{model_name}")

    vec_env = model.get_env()
    obs = vec_env.reset()

    for i in range(5000):
        action, _state = model.predict(obs, deterministic=False)
        obs, _reward, _done, _info = vec_env.step(action)
        print(_info)
        vec_env.render()
        time.sleep(1/20)

# Models are saved in folder "models"
if __name__ == "__main__":
    run_model(pretrained=True, model_name="mario_rl")

