import gymnasium as gym
import multiprocessing

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

timesteps = 5_000_000


# multiprocessing
def make_env(rank, seed=0):
    def _make():
        env = gym.make("CarRacing-v2", render_mode="rgb_array")
        env.reset(seed=seed + rank)
        return env

    return _make


def start():
    nproc = multiprocessing.cpu_count()

    env = SubprocVecEnv([make_env(i) for i in range(nproc)])

    # cnn policy because it's vision!
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./log/")
    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save("out/selfdrivingcar")


if __name__ == "__main__":
    start()
