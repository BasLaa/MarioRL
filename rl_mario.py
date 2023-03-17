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

N_TIMESTEPS = 2000000
LEARNING_RATE = 0.00001
GAMMA = 0.99
N_EPOCHS = 10
N_STEPS = 4096
BATCH_SIZE = 128

SKIP_FREQ = 6

# Save a model every 'LOG_FREQ' timesteps
LOG_FREQ = N_TIMESTEPS // 30

# Linear decrease of learning rate with progress
def lr_schedule(initial_value):
    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0.
        """
        return progress * initial_value

    return func

def run_model(env, pretrained=False, model_name="mario_rl", callback=None, logger=None):

    if pretrained and os.path.isfile(f'models/{model_name}/{model_name}.zip'):
        print("Found existing model...")
        model = PPO.load(f'models/{model_name}/{model_name}', env=env)
    elif pretrained:
        print("Model not found...")
        return
    else:
        print("Training new model...")

        model = PPO(
            "CnnPolicy", env, verbose=1, learning_rate=LEARNING_RATE, n_steps=N_STEPS
          , gamma=GAMMA, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,)
        
        model.set_logger(logger)
        with callbacks.ProgressBarManager(N_TIMESTEPS) as progress_callback:
            final_callback = CallbackList([callback, progress_callback])
            model.learn(total_timesteps=N_TIMESTEPS, callback=final_callback)
        model.save(f"models/{model_name}/{model_name}")

    done = True
    for i in range(5000):
        if done:
            obs = env.reset()
        action, _state = model.predict(obs, deterministic=False)
        obs, _reward, done, _info = env.step(action)
        print(_info)
        print(_reward)
        env.render()
        time.sleep(1/60)


# Models are saved in folder "models"
if __name__ == "__main__":

    env = gym_super_mario_bros.make('SuperMarioBros-v0')

    # skip is number of frames that are skipped
    env = make_env.make_env(env, skip=SKIP_FREQ, move_set=RIGHT_ONLY)

    # Stops training on no improvement (Not using atm)
    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=30, min_evals=5, verbose=1)
    # eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)

    model_name = "night_2_2m"

    checkpoint_callback = callbacks.TrainAndLoggingCallback(
        log_freq=LOG_FREQ,
        save_path=f"./models/{model_name}/model_logs",
        name_prefix=model_name,
    )

    logger_path = f"./models/{model_name}/logs/"
    # set up logger
    new_logger = configure(logger_path, ["csv", "tensorboard"])


    # TRAINING MODEL
    run_model(env, model_name=model_name, pretrained=False, callback=checkpoint_callback, logger=new_logger)

    # RUNNING TRAINED MODEL
    # run_model(env=env, pretrained=True, model_name=model_name)