import gym
import numpy as np
from Agent import ActorCritic_Agent
import torch
import torch.optim as optim

# PARAMETERS if not initialized from Experiment.py
num_iterations = 1000
num_eval_episodes = 10
eval_interval = 100

batch_size = 64
learning_rate = 1e-3
gamma = 0.99
beta = 0.01

def actorcritic(n_timesteps=num_iterations, learning_rate=learning_rate, gamma=gamma, 
              eval_interval=eval_interval, render_mode="human",beta=beta):
    
    env = gym.make("LunarLander-v2", render_mode=render_mode)
    env_eval = gym.make("LunarLander-v2")

    actorCriticAgent = ActorCritic_Agent(n_states=8, n_actions=4, learning_rate=learning_rate,
                              gamma=gamma, beta=beta)
    
    optimizer = optim.Adam(actorCriticAgent.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    eval_timesteps = []
    eval_returns = []
    
    loss = 0
    iteration = 0
    episode = 0
    while iteration < n_timesteps:
        state, info = env.reset(seed=42) 
        episode_rewards = []
        log_probs = []
        state_values = []
        entropies = []
        terminated = False
        truncated = False
        
        while  not (terminated or truncated):
            # action, log_prob, entropy = actorCriticAgent.select_action(state)
            action = actorCriticAgent(state)
            observation, reward, terminated, truncated, info = env.step(action)
            
            actorCriticAgent.rewards.append(reward)
            
            # episode_rewards.append(reward)
            # log_probs.append(log_prob)
            
            # _, state_value = actorCriticAgent(state)
            # state_values.append(state_value)
            # entropies.append(entropy)
            
            state = observation
            iteration += 1
            
            # Evaluation step
            if iteration % eval_interval == 0:
                    eval_timesteps.append(iteration)
                    eval_returns.append(actorCriticAgent.evaluate(env_eval, n_eval_episodes=num_eval_episodes))
                    print(f"Step: {iteration}, Average Return: {np.mean(eval_returns[-1])}")
                    print(f"Total episodes: {episode}")
                    # print(f"loss = {loss}")
        
            if iteration >= n_timesteps:
                break
            if terminated or truncated:
                episode += 1
                break
        
        optimizer.zero_grad()
        loss = actorCriticAgent.calculateLoss(gamma)
        loss.backward()
        optimizer.step()        
        actorCriticAgent.clearMemory()       
        
    # del actorCriticAgent.actor
    # del actorCriticAgent.critic
    
    env.close()
    env_eval.close()
    
    return np.array(eval_returns), np.array(eval_timesteps)

if __name__ == '__main__':
    returns, timesteps = actorcritic()
