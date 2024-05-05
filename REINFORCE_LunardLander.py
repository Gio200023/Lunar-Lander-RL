import gym
import numpy as np
from Agent import REINFORCEAgent
import torch
import sys

num_iterations = 1000 
num_eval_episodes = 10 
eval_interval = 1000  

initial_collect_steps = 100  
collect_steps_per_iteration =   1
replay_buffer_max_length = 10000

batch_size = 64  
log_interval = 200

learning_rate = 1e-3  
gamma = 0.99
epsilon = 0.05
temp = 0.05

def reinforce(n_timesteps=num_iterations, learning_rate=learning_rate, gamma=gamma, 
                eval_interval=eval_interval, render_mode = "rgb_array", beta=0.1):
    
    env = gym.make("LunarLander-v2",render_mode=render_mode, continuous = False,gravity = -10.0,enable_wind = False)
    env_eval = gym.make("LunarLander-v2", continuous = False,gravity = -10.0,enable_wind = False)
    
    reinforceAgent = REINFORCEAgent(
                        n_states=8, 
                        n_actions=4, 
                        learning_rate=learning_rate, 
                        gamma=gamma)
                        
    observation, info = env.reset(seed=42) 

    eval_timesteps = []
    eval_returns = []
    max_min = []

    iteration = 0
    episode = 0
    
    while iteration <= n_timesteps:
        episode_rewards = []
        log_probs = []
        entropies = 0

        state, info = env.reset()

        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action, log_prob, entropy = reinforceAgent.select_action(state)
            observation, reward, terminated, truncated, info = env.step(action)

            episode_rewards.append(reward)
            log_probs.append(log_prob)
            entropies += entropy
            
            state = observation 
            
            if iteration % eval_interval == 0:
                eval_timesteps.append(iteration)
                eval_ret,maxim,minim = reinforceAgent.evaluate(env_eval, n_eval_episodes=num_eval_episodes)
                eval_returns.append(eval_ret)
                max_min.append([maxim,minim])
                print(f"(reinforce) step: {iteration}, Average Reward : {eval_ret}")

            iteration+=1
            reinforceAgent._current_iteration=iteration

            if iteration >= n_timesteps:
                break
            if terminated or truncated:
                episode += 1
                break
        
        reinforceAgent.update_policy_network(rewards=episode_rewards, log_probs=log_probs, entropies=entropies, beta=beta)
    
    del reinforceAgent.policy_network
    env.close()
    
    return np.array(eval_returns), np.array(eval_timesteps), np.array(max_min)


if __name__ == '__main__':
    reinforce()