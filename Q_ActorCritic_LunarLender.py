import gym
import numpy as np
from Agent import Q_ActorCritic_Agent
import torch
import torch.optim as optim
import torch.distributions as distributions
import sys

# PARAMETERS if not initialized from Experiment.py
num_iterations = 1000
num_eval_episodes = 10
eval_interval = 100

batch_size = 64
learning_rate = 1e-3
gamma = 0.99
beta = 0.01

def actorcritic(n_timesteps=num_iterations, learning_rate=learning_rate, gamma=gamma, 
              eval_interval=eval_interval, render_mode="human", update="both"):
    
    env = gym.make("LunarLander-v2", render_mode=render_mode)
    env_eval = gym.make("LunarLander-v2")

    actorCriticAgent = Q_ActorCritic_Agent(n_states=8, n_actions=4, learning_rate=learning_rate,
                              gamma=gamma)

    eval_timesteps = []
    eval_returns = []
    
    iteration = 0
    episode = 0
    while iteration < n_timesteps:
        episode_rewards = []
        log_probs = []
        values = []
        entropy = 0
        terminated = False
        truncated = False
        
        state, info = env.reset(seed=42) 
        
        while  not (terminated or truncated):

            value, policy_dist = actorCriticAgent.forward(state)

            if np.random.rand() < 0.1:  # Epsilon-greedy for exploration
                action = env.action_space.sample()
            else:
                action = torch.multinomial(policy_dist.squeeze(), num_samples=1).item()

            log_prob = torch.log(policy_dist.squeeze(0)[action])
            
            # action, log_prob = actorCriticAgent.select_action_(policy_dist)
            new_entropy = -torch.sum(torch.mean(policy_dist) * torch.log(policy_dist))
            next_state, reward, terminated, truncated, info = env.step(action)
            
            episode_rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            entropy += new_entropy 
            
            iteration += 1
            actorCriticAgent._current_iteration = iteration
            state = next_state
            
            # Evaluation step
            if iteration % eval_interval == 0:
                    eval_timesteps.append(iteration)
                    eval_returns.append(actorCriticAgent.evaluate(env_eval, n_eval_episodes=num_eval_episodes))
                    print(f"Step: {iteration}, Average Return: {np.mean(eval_returns[-1])}")
                    print(f"Total episodes: {episode}")
        
            if iteration >= n_timesteps:
                break
            if terminated or truncated:
                episode += 1
                break
        
        # UPDATE PHASE
        n_step = 10  
        total_steps = len(episode_rewards)  

        for start_index in range(total_steps - n_step + 1):
            end_index = start_index + n_step

            current_rewards = episode_rewards[start_index:end_index]
            current_log_probs = log_probs[start_index:end_index]
            current_values = values[start_index:end_index]

            if update == "td":
                actorCriticAgent.update_td(state, next_state, action, reward, entropy)  
            elif update == "both":
                actorCriticAgent.update_both(state, current_rewards, current_log_probs, current_values, entropy)   
            elif update == "base":
                actorCriticAgent.update_baseline_only(state, current_rewards, current_log_probs, current_values, entropy)   
            elif update == "boot":
                actorCriticAgent.update_bootstrap_only(state, current_rewards, current_log_probs, current_values, entropy)
 
        if total_steps > n_step:
            for start_index in range(total_steps - n_step, total_steps):
                current_rewards = episode_rewards[start_index:total_steps]
                current_log_probs = log_probs[start_index:total_steps]
                current_values = values[start_index:total_steps]

                if update == "both":
                    actorCriticAgent.update_both(state, current_rewards, current_log_probs, current_values, entropy)   
                elif update == "base":
                    actorCriticAgent.update_baseline_only(state, current_rewards, current_log_probs, current_values, entropy)   
                elif update == "boot":
                    actorCriticAgent.update_bootstrap_only(state, current_rewards, current_log_probs, current_values, entropy)

          
            
    del actorCriticAgent.actor
    del actorCriticAgent.critic
    
    env.close()
    env_eval.close()
    
    print(f"Total number of episodes {episode}")
    return np.array(eval_returns), np.array(eval_timesteps)

if __name__ == '__main__':
    returns, timesteps = actorcritic()



