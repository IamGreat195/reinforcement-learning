import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, render=False, is_training=True):
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human" if render else None, map_name="8x8", )

    if(is_training):
      q = np.zeros((env.observation_space.n, env.action_space.n))  # Initialize Q-table with zeros
    else:
      f = open('frozenlakeQ1_rewards.pkl', 'rb')
      q = pickle.load(f)
      f.close()

    learning_rate = 0.9
    discount_factor = 0.9

    rewards_per_episode = np.zeros(episodes)  # Store rewards for each episode

    epsilon = 1 # Exploration rate
    epsilon_decay = 0.0001 # epsilon decay rate 1/0.0001 = 10000 episodes
    rng = np.random.default_rng()

    for i in range(episodes):
      state = env.reset()[0]
      terminated = False # True when fall in hole or reach goal
      truncated = False # True when actions > 200

      while(not terminated and not truncated):
          if is_training and rng.random() < epsilon:
              action = env.action_space.sample() # Explore: select a random action  0: left, 1: down, 2: right, 3: up
          else:
              action = np.argmax(q[state, :])

          new_state, reward, terminated, truncated, info = env.step(action)

          if is_training:
              q[state, action] = q[state, action] + learning_rate * (reward + discount_factor * np.max(q[new_state]) - q[state, action]) 
          state = new_state

      epsilon = max(0, epsilon - epsilon_decay)  # Decay exploration rate

      if(epsilon==0):
          learning_rate = 0.001  # Reduce learning rate to allow fine-tuning
      
      if reward == 1:
          rewards_per_episode[i] = 1


    env.close()

    sum_rewards = np.zeros(episodes)
    for i in range(episodes):
        sum_rewards[i] = np.sum(rewards_per_episode[max(0, i-100):i+1])
    plt.plot(sum_rewards)
    plt.savefig("frozenlakeQ1_rewards.png")

    if is_training:
      f = open('frozenlakeQ1_rewards.pkl', 'wb')
      pickle.dump(q, f)
      f.close()  

3

if __name__ == "__main__":
    run(15000, is_training=False, render=True) 

