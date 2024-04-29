import gym
import numpy as np
from Agent import Q_ActorCritic_Agent
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

    actorCriticAgent = Q_ActorCritic_Agent(n_states=8, n_actions=4, learning_rate=learning_rate,
                              gamma=gamma, beta=beta)
    
    optimizer = optim.Adam(actorCriticAgent.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    eval_timesteps = []
    eval_returns = []
    
    n_states=8
    
    loss = 0
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

            value, policy_dist = actorCriticAgent(state)

            action = torch.multinomial(policy_dist.squeeze(), num_samples=1).item()
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            new_entropy = -torch.sum(torch.mean(policy_dist) * torch.log(policy_dist))
            next_state, reward, terminated, truncated, info = env.step(action)
            
            episode_rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            new_entropy += entropy
            
            state = next_state
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
        episode_rewards = torch.tensor(episode_rewards, dtype=torch.float32)
        # cslculate target temporal difference at 0
        next_value = actorCriticAgent.critic(torch.tensor(state, dtype=torch.float32))
        td_target = episode_rewards + gamma * next_value
        
        # compute critic loss and update 
        critic_loss = torch.mean((td_target - actorCriticAgent.critic(state)) ** 2)
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        
        # compute advantge and update actor
        advantages = td_target - actorCriticAgent.critic(state)
        actor_loss = -torch.mean(log_probs * advantages.detach())
        optimizer.zero_grad()
        actor_loss.backward()
        optimizer.step()
        
        actorCriticAgent.update(state, episode_rewards, log_probs, values, entropy)   
        
        
    del actorCriticAgent.actor
    del actorCriticAgent.critic
    
    env.close()
    env_eval.close()
    
    return np.array(eval_returns), np.array(eval_timesteps)

if __name__ == '__main__':
    returns, timesteps = actorcritic()


