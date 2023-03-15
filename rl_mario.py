import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure

import make_env

import time 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, name_prefix="", verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, self.name_prefix+'_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

def run_model(env, pretrained=False, model_name="mario_rl", callback=None, logger=None):

    if pretrained and os.path.isfile(f'models/{model_name}.zip'):
        print("Found existing model...")
        model = PPO.load(f'models/{model_name}', env=env)
    elif pretrained:
        print("Model not found...")
        return
    else:
        print("Training new model...")
        model = PPO("MlpPolicy", env, tensorboard_log="./tensorboard_log/", verbose=1, n_epochs=10, n_steps=2048, batch_size=64)
        model.set_logger(logger)
        model.learn(total_timesteps=15000, callback=callback)
        model.save(f"models/{model_name}")

    vec_env = model.get_env()
    obs = vec_env.reset()

    for i in range(5000):
        action, _state = model.predict(obs, deterministic=False)
        obs, _reward, _done, _info = vec_env.step(action)
        vec_env.render()
        time.sleep(1/120)


# Models are saved in folder "models"
if __name__ == "__main__":

    env = gym_super_mario_bros.make('SuperMarioBros-v0')

    # skip is number of frames that are skipped
    env = make_env.make_env(env, skip=4, move_set=RIGHT_ONLY)

    # Stops training on no improvement
    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=30, min_evals=5, verbose=1)
    # eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)

    checkpoint_callback = TrainAndLoggingCallback(
        check_freq=5000,
        save_path="./model_logs/",
        name_prefix="ppo_model",
    )

    logger_path = "./logs/"
    # set up logger
    new_logger = configure(logger_path, ["stdout", "csv", "tensorboard"])


    run_model(env, model_name="ppo_model_total", pretrained=False, callback=checkpoint_callback, logger=new_logger)