import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure

import callbacks
import make_env

import time
import os
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

N_TIMESTEPS = 3000000
LEARNING_RATE = 0.0001
GAMMA = 0.99
N_EPOCHS = 10
N_STEPS = 2048
BATCH_SIZE = 64

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


def run_model(env, pretrained=False, continue_learning=False, model_name="mario_rl", callback=None, logger=None):

    if pretrained:
        files = glob.glob(f"./**/{model_name}.zip", recursive = True)
        if len(files) == 1:
            print("Found existing model...")
            model = PPO.load(files[0], env=env)

            if continue_learning:
                model.set_logger(logger)
                model.set_env(env)
                with callbacks.ProgressBarManager(N_TIMESTEPS) as progress_callback:
                    final_callback = CallbackList([callback, progress_callback])
                    model.learn(total_timesteps=N_TIMESTEPS, callback=final_callback)
                model.save(f"models/{model_name}/{model_name}")
        else:
            print("Model not found...")
            return
    else:
        print("Training new model...")
        model = PPO(
            "CnnPolicy", env, verbose=1, learning_rate=LEARNING_RATE, n_steps=N_STEPS
            , gamma=GAMMA, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, )

        model.set_logger(logger)
        with callbacks.ProgressBarManager(N_TIMESTEPS) as progress_callback:
            final_callback = CallbackList([callback, progress_callback])
            model.learn(total_timesteps=N_TIMESTEPS, callback=final_callback)
        model.save(f"models/{model_name}/{model_name}")

    done = True
    obs = env.reset()
    for i in range(5000):
        if done:
            obs = env.reset()
        action, _state = model.predict(obs, deterministic=False)
        obs, _reward, done, _info = env.step(int(action))
        # print(_info)
        # print(_reward)
        env.render()
        time.sleep(1 / 120)


# Models are saved in folder "models"
if __name__ == "__main__":
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    # env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # skip is number of frames that are skipped
    env = make_env.make_env(env, skip=SKIP_FREQ, move_set=RIGHT_ONLY)

    # Stops training on no improvement (Not using atm)
    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=30, min_evals=5, verbose=1)
    # eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)

    # model_name = "nihao_124995"
    model_name = "nikola_baseline_5_133333"

    checkpoint_callback = callbacks.TrainAndLoggingCallback(
        log_freq=LOG_FREQ,
        save_path=f"./models/{model_name}/model_logs",
        name_prefix=model_name,
    )

    logger_path = f"./models/{model_name}/logs/"
    # TRAINING MODEL
    
    # set up logger (ONLY UNCOMMENT IF TRAINING)
    # new_logger = configure(logger_path, ["stdout", "csv", "tensorboard"])

    # run_model(env, model_name=model_name, pretrained=False, callback=checkpoint_callback, logger=new_logger)

    # CONTINUE TRAINING MODEL
    # run_model(env=env, pretrained=True, continue_learning=True, model_name=model_name, callback=checkpoint_callback,
            #   logger=new_logger)

    # RUNNING TRAINED MODEL
    run_model(env=env, pretrained=True, model_name=model_name)
