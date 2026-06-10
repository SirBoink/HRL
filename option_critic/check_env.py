import gymnasium as gym
import mo_gymnasium as mo_gym
env = gym.make('four-room-v0', disable_env_checker=True)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
state, _ = env.reset()
print("Initial state:", state)
print("State type:", type(state))
