#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for master course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Helper import softmax, argmax
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions import Categorical

class ActorCritic_Agent(nn.Module):
    """
        Actor-critic Agent class

    Returns:
        int: best action according to the policy
    """
    def __init__(self, n_states, n_actions, learning_rate, gamma, _entropy):
        super(ActorCritic_Agent, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma  
        self.learning_rate = learning_rate
        self._current_iteration = 0
        self._entropy = _entropy
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        self.to(self.device)

    def forward(self, x, target=False):
        state = torch.FloatTensor(state).to(self.device)
        action_probs = F.softmax(self.actor(state), dim=-1)
        state_value = self.critic(state)
        return action_probs, state_value
        
    def select_action(self, state):
        action_probs, _ = self(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, entropy

    def update(self, rewards, log_probs, state_values, entropies):
            discounts = [self.gamma ** i for i in range(len(rewards))]
            returns = [sum(discounts[:len(rewards)-i] * rewards[i:]) for i in range(len(rewards))]
            returns = torch.tensor(returns).to(self.device)

            loss = 0
            for log_prob, value, R, entropy in zip(log_probs, state_values, returns, entropies):
                advantage = R - value.item()
                actor_loss = -(log_prob * advantage) - self._entropy * entropy
                critic_loss = F.mse_loss(torch.tensor([R]).to(self.device), value)
                loss += actor_loss + critic_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate(self, env, n_eval_episodes=30, max_episode_length=500):
        total_rewards = []
        for _ in range(n_eval_episodes):
            state = env.reset()
            episode_reward = 0
            for _ in range(max_episode_length):
                action, _, _ = self.select_action(state)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                if done:
                    break
            total_rewards.append(episode_reward)
        return np.mean(total_rewards)