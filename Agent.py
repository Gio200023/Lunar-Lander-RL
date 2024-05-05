# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Practical for master course 'Reinforcement Learning',
# Leiden University, The Netherlands
# By Thomas Moerland
# """

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import time

class REINFORCEAgent(nn.Module):
    """
    Reinforce Agent class
        
    Returns:
        int: best action according to the policy
    """
    def __init__(self, n_states, n_actions, learning_rate, gamma):
        super(REINFORCEAgent, self).__init__()
        self.n_actions = n_actions
        self.n_states = n_states
        self.gamma = gamma
        self.learning_rate = learning_rate
        self._current_iteration = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("using: " + str(self.device)+ " device")
        
        self.policy_network = nn.Sequential(
            nn.Linear(self.n_states, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions),
            nn.Softmax(dim=-1)
        )
        
        for layer in self.policy_network:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0)
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        self.to(self.device)

    def forward(self, x):
        return self.policy_network(x)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probabilities = self.policy_network(state)

        try:
            action = torch.multinomial(probabilities, 1).item()
        except RuntimeError as e:
            print("Failed to sample action on GPU. Falling back to CPU. Error:", e)

            probabilities_cpu = probabilities.cpu()
            action = torch.multinomial(probabilities_cpu, 1).item()

        log_prob = torch.log(probabilities.squeeze(0)[action])
        entropy = -(probabilities * torch.log(probabilities)).sum()
        return action, log_prob, entropy
    
    def update_policy_network(self, rewards, log_probs, entropies, beta=0.1):
        
        discounted_rewards = []
        cum_reward = 0
        for reward in rewards[::-1]:
            cum_reward = reward + self.gamma * cum_reward
            discounted_rewards.insert(0, cum_reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.stack(policy_loss).sum()
        
        entropy_loss = torch.mean(entropies)
        policy_loss = policy_loss - beta * entropy_loss
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
    def evaluate(self, eval_env, n_eval_episodes=30, max_episode_length=200):
        total_rewards = []
        episode_maxes = []
        episode_mins = []
        
        for _ in range(n_eval_episodes):
            state, info = eval_env.reset()
            episode_rewards = []
            done = False
            truncated = False
            iteration = 0
            
            while not done and not truncated and iteration < max_episode_length:
                action, log_prob, entropy = self.select_action(state)
                state, reward, done, truncated, info = eval_env.step(action)
                episode_rewards.append(reward)
                iteration += 1
            
            total_rewards.append(sum(episode_rewards))
            episode_maxes.append(max(episode_rewards))
            episode_mins.append(min(episode_rewards))
        
        mean_return = np.mean(total_rewards)
        return mean_return, np.mean(episode_maxes), np.mean(episode_mins)