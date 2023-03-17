import gym
import numpy as np
import collections
import cv2
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=4)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init to first obs"""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    """
    Downsamples image to 84x84
    Greyscales image

    Returns numpy array
    """
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)

class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize pixel values in frame --> 0 to 1"""
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class CustomRewardEnv(gym.Wrapper):
    def __init__(self, env, max_wait=40, kill=True):
        super().__init__(env)
        self.current_x = 0
        self.fault_counter = 0
        self.max_wait = max_wait
        self.kill = kill
        
    def reset(self, **kwargs):
        self.current_x = 0
        self.count = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        new_x = info['x_pos']
        if new_x <= self.current_x:
            self.fault_counter += 1
        else:
            self.fault_counter = 0
            self.current_x = new_x
        
        if self.fault_counter >= self.max_wait:
            reward = -15
            if self.kill:
                done = True

        if done:
            if info["flag_get"]:
                reward = 50
        
        if info['world'] == 1:
            if info['y_pos'] < 75:
                reward = -10
        
        return state, reward, done, info

def make_env(env, skip=4, move_set=RIGHT_ONLY):
    env = MaxAndSkipEnv(env, skip=skip)
    env = ProcessFrame84(env)
    env = ScaledFloatFrame(env)
    env = CustomRewardEnv(env)
    env = JoypadSpace(env, move_set)
    return env