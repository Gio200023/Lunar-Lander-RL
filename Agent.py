#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for master course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import torch, sys
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torch.distributions as distributions

class Q_ActorCritic_Agent(nn.Module):
    """
        Actor-critic Agent class

    Returns:
        int: best action according to the policy
    """
    def __init__(self, n_states, n_actions, learning_rate, gamma, beta):
        super(Q_ActorCritic_Agent, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma  
        self.learning_rate = learning_rate
        self._current_iteration = 0
        self.beta = beta
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logprobs = []
        self.state_values = []
        self.rewards = []

        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        
        #initialize actor network
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0)
        
        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
         #initialize critic network
        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        self.to(self.device)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        value = self.critic(state)
        policy_dist = F.softmax(self.actor(state), dim=1)
        
        return value, policy_dist

    def select_action_(self,policy_dist):
        dist = distributions.Categorical(policy_dist)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    
    def update_both(self,state, rewards, log_probs, values, entropy):
        
        values = torch.tensor(values,dtype=torch.float32).to(self.device)
        
        Qvals = torch.zeros_like(values)
        q_val, _ = self.forward(state)
        for t in reversed(range(len(rewards))):
            q_val = rewards[t] + self.gamma * q_val
            Qvals[t] = q_val
        
        log_probs = torch.stack(log_probs)
        advantages = Qvals - values
        baseline = values.mean()
        advantages -= baseline  
        
        actor_loss = (-log_probs *advantages).mean()
        critic_loss = 0.5 * advantages.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy
        

        self.optimizer.zero_grad()
        ac_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # print("log_probs",log_probs)
        # print("advantages",advantages)
        # print("baseline",baseline)
        # print("actor_loss",actor_loss)
        # print("critic_loss",critic_loss)
        # print("ac_loss",ac_loss)
        # print("rewards",rewards)
        
        # if self._current_iteration > 200:
        #     sys.exit(0)
        
        
    def update_bootstrap_only(self, state, rewards, log_probs, values, entropy):
        
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        
        Qvals = torch.zeros_like(values)
        q_val, _ = self.forward(state)
        for t in reversed(range(len(rewards))):
            q_val = rewards[t] + self.gamma * q_val
            Qvals[t] = q_val
        
        log_probs = torch.stack(log_probs)
        advantages = Qvals - values  # Only bootstrapping, no baseline subtraction
        
        actor_loss = (-log_probs * advantages).mean()
        critic_loss = 0.5 * advantages.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy

        self.optimizer.zero_grad()
        ac_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

    def update_baseline_only(self, state, rewards, log_probs, values, entropy):
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        
        baseline = values.mean()
        advantages = values - baseline  # Only using baseline to calculate advantages
        
        log_probs = torch.stack(log_probs)
        actor_loss = (-log_probs * advantages).mean()
        critic_loss = 0.5 * advantages.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy

        self.optimizer.zero_grad()
        ac_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

    def evaluate(self, env, n_eval_episodes=30, max_episode_length=200):
        total_rewards = []
        
        for _ in range(n_eval_episodes):
            state, info = env.reset()
            episode_rewards = []
            done = False
            truncated = False
            iteration = 0
            
            while not done and not truncated and iteration < max_episode_length:
                _, policy_dist = self(state)
                action = torch.multinomial(policy_dist.squeeze(), num_samples=1).item()
                state, reward, done, truncated, info = env.step(action)
                episode_rewards.append(reward)
                iteration += 1
            
            total_rewards.append(sum(episode_rewards))
        
        mean_return = np.mean(total_rewards)
        return mean_return