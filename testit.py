import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CarRacing-v2", render_mode="human")
model = PPO("CnnPolicy", env, verbose=1)

print("Loading model...")
model.set_parameters("./out/complete.zip")
print("Done!")

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(2000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
