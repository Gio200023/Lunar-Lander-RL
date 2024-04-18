import gym
import numpy as np
from Agent import REINFORCEAgent

# PARAMETERS if not initialized from Experiment.py
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
    
    reinforceAgent = REINFORCEAgent(n_states=8, 
                        n_actions=4, 
                        learning_rate=learning_rate, 
                        gamma=gamma)
                        
    observation, info = env.reset(seed=42) 

    eval_timesteps = []
    eval_returns = []

    iteration = 0
    while iteration <= n_timesteps:
        episode_rewards = []
        log_probs = []
        entropies = []
        
        state, info = env.reset()
        episode_rewards.clear()
        log_probs.clear()

        terminated = False
        while not terminated:
            action, log_prob, entropy = reinforceAgent.select_action(state)
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_rewards.append(reward)
            log_probs.append(log_prob)
            entropies.append(entropy)
            
            state = observation 
            
            if iteration % eval_interval == 0:
                eval_timesteps.append(iteration)
                eval_returns.append(reinforceAgent.evaluate(env_eval, n_eval_episodes=num_eval_episodes))
                print("step: ",iteration)

            iteration+=1
            reinforceAgent._current_iteration=iteration

            if iteration >= n_timesteps:
                break
            if terminated:
                break
            
            # if iteration % eval_interval == 0:
            #         print("episode_rewards = " + str(episode_rewards))
            #         print("log_probs = " + str(log_probs))
            
        
        returns = []
        G = 0
        for reward in reversed(episode_rewards):
            G = reward + gamma * G
            returns.insert(0, G)
            
        reinforceAgent.update_policy_network(rewards=returns, log_probs=log_probs, entropies=entropies, beta=beta)
        
    env.close()
    
    return np.array(eval_returns), np.array(eval_timesteps) 


if __name__ == '__main__':
    reinforce()