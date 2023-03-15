# # Number of training trials
# N_TRIALS = 1
# # Timeout time for optimization in seconds
# TIMEOUT = 6000

# def sample_params(trial):
#     """ Returns the parameters we want to optimize """
#     return {
#         'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
#         'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
#         'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
#     }

# def objective(trial):
#     model_params = sample_params(trial)
#     env = gym_super_mario_bros.make('SuperMarioBros-v0')

#     # skip is number of frames that are skipped
#     env = make_env.make_env(env, skip=4, move_set=RIGHT_ONLY)

#     model = PPO('MlpPolicy', env, verbose=1, batch_size=model_params['n_steps']//4)
#     model.learn(total_timesteps=5000)

#     mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)

#     return mean_reward