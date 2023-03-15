import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure

import callbacks
import make_env

import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 4 million steps
N_TIMESTEPS = 4000000
LEARNING_RATE = 0.0001
GAMMA = 0.95
N_EPOCHS = 40
N_STEPS = 2048
BATCH_SIZE = 64

# Save a model every 'LOG_FREQ' timesteps (10 models here)
LOG_FREQ = N_TIMESTEPS // 10


def run_model(env, pretrained=False, model_name="mario_rl", callback=None, logger=None):
    if pretrained and os.path.isfile(f'models/{model_name}.zip'):
        print("Found existing model...")
        model = PPO.load(f'models/{model_name}', env=env)
    elif pretrained:
        print("Model not found...")
        return
    else:
        print("Training new model...")
        model = PPO("CnnPolicy", env, verbose=1, learning_rate=LEARNING_RATE, gamma=GAMMA, n_epochs=N_EPOCHS,
                    n_steps=N_STEPS, batch_size=BATCH_SIZE)
        model.set_logger(logger)
        with callbacks.ProgressBarManager(
                N_TIMESTEPS) as progress_callback:  # this the garanties that the tqdm progress bar closes correctly
            final_callback = CallbackList([callback, progress_callback])
            model.learn(total_timesteps=N_TIMESTEPS, callback=final_callback)
        model.save(f"models/{model_name}")

    vec_env = model.get_env()
    obs = vec_env.reset()

    for i in range(5000):
        action, _state = model.predict(obs, deterministic=False)
        obs, _reward, _done, _info = vec_env.step(action)
        vec_env.render()
        time.sleep(1 / 120)


# Models are saved in folder "models"
if __name__ == "__main__":
    env = gym_super_mario_bros.make('SuperMarioBros-v0')

    # skip is number of frames that are skipped
    env = make_env.make_env(env, skip=4, move_set=RIGHT_ONLY)

    # Stops training on no improvement (Not using atm)
    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=30, min_evals=5, verbose=1)
    # eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)

    checkpoint_callback = callbacks.TrainAndLoggingCallback(
        log_freq=LOG_FREQ,
        save_path="./model_logs/",
        name_prefix="ppo_model",
    )

    logger_path = "./logs/"
    # set up logger
    new_logger = configure(logger_path, ["stdout", "csv", "tensorboard"])

    # TRAINING MODEL
    run_model(env, model_name="ppo_model_total", pretrained=False, callback=checkpoint_callback, logger=new_logger)

    # RUNNING TRAINED MODEL
    # run_model(env=env, pretrained=True, model_name="ppo_model_total")
