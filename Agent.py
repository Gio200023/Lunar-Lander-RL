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
            nn.Linear(self.n_states, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_actions),
            nn.Softmax(dim=-1)
        )
        #initialize policy network
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
        # if self._current_iteration % 500 == 0:
        #     print(probabilities)
        action = torch.multinomial(probabilities, 1).item()
        log_prob = torch.log(probabilities.squeeze(0)[action])
        entropy = -(probabilities * torch.log(probabilities)).sum()
        # print("\n")
        # print("prob:" +str(probabilities))
        # print("action:" +str(action))
        # print("\n")
        return action, log_prob, entropy

    
    def update_policy_network(self, rewards, log_probs, entropies, beta=0.1):
        rewards = np.array(rewards)
        discounts = np.power(self.gamma, np.arange(len(rewards)))
        returns = np.array([np.sum(rewards[i:] * discounts[:len(rewards)-i]) for i in range(len(rewards))])
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        policy_loss = []
        entropy_term = []
        for log_prob, R, entropy in zip(log_probs, returns, entropies):
            loss_item = -log_prob * R  
            policy_loss.append(loss_item.unsqueeze(0))  
            entropy_term.append(entropy.unsqueeze(0))  

        policy_loss = torch.cat(policy_loss).sum()  
        entropy_loss = torch.cat(entropy_term).sum() * beta 

        total_loss = policy_loss + entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()
    
        
    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=300):
        total_rewards = []
        for _ in range(n_eval_episodes):
            state, info = eval_env.reset()
            episode_reward = 0
            done = False
            iteration=0
            while not done and iteration < max_episode_length:
                action, log_prob, entropy = self.select_action(state)
                state, reward, done, truncated, info = eval_env.step(action)
                episode_reward += reward
                iteration+=1
            total_rewards.append(episode_reward)
        mean_return = np.mean(total_rewards)
        return mean_return