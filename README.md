# MarioRL
Intelligent Systems project on reinforcement learning for Super Mario

RL Library: https://intellabs.github.io/coach/index.html

Super Mario API: https://pypi.org/project/gym-super-mario-bros/ is based on Gym: https://www.gymlibrary.dev/api/core/


## Installation
- Install the Mario API with `pip install gym-super-mario-bros` and `nes-py` with `pip install nes-py`

- Run from command line as `gym_super_mario_bros -e <the environment ID to play> -m <human or random>` <br>
  NOTE: by default, -e is set to SuperMarioBros-v0 and -m is set to human. <br>
  NOTE: This doesn't work for me

- Run the example runner file instead to play as human: `python run_mario.py`

- If you get an error `ValueError: not enough values to unpack (expected 5, got 4)` go to `.../gym/wrappers/time_limit.py` and remove `truncated` everywhere
