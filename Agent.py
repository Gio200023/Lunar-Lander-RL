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
    def __init__(self, n_states, n_actions, learning_rate, gamma, beta):
        super(ActorCritic_Agent, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma  
        self.learning_rate = learning_rate
        self._current_iteration = 0
        self.beta = beta
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.affine = nn.Linear(8, 128)
        
        self.action_layer = nn.Linear(128, 4)
        self.value_layer = nn.Linear(128, 1)
        
        self.logprobs = []
        self.state_values = []
        self.rewards = []

        # # Actor Network
        # self.actor = nn.Sequential(
        #     nn.Linear(n_states, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, n_actions)
        # )
        
        # #initialize actor network
        # for layer in self.actor:
        #     if isinstance(layer, nn.Linear):
        #         init.xavier_uniform_(layer.weight)
        #         init.constant_(layer.bias, 0)
        
        # # Critic Network
        # self.critic = nn.Sequential(
        #     nn.Linear(n_states, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1)
        # )
        
        #  #initialize critic network
        # for layer in self.critic:
        #     if isinstance(layer, nn.Linear):
        #         init.xavier_uniform_(layer.weight)
        #         init.constant_(layer.bias, 0)
        
        # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        self.to(self.device)

    # def forward(self, state):
    #     state = torch.FloatTensor(state).to(self.device)
    #     action_probs = F.softmax(self.actor(state), dim=-1)
    #     state_value = self.critic(state)
    #     return action_probs, state_value
    
    def forward(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        state = F.relu(self.affine(state))
        
        state_value = self.value_layer(state)
        
        action_probs = F.softmax(self.action_layer(state))
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        return action.item()
        
    def select_action(self, state):
        action_probs, _ = self(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, entropy
    
    def calculateLoss(self, gamma=0.99):
        
        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        # normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            # print("reward ",len(reward))
            # print("value ",len(value))
            value_loss = F.mse_loss(reward,value)
            loss += (action_loss + value_loss)   
        return loss
    
    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]

    def update(self, rewards, log_probs, state_values, entropies):
            rewards = np.array(rewards)
            discounts = [self.gamma ** i for i in range(len(rewards))]
            # print("discounts "+str(discounts))
            # print("rewards "+str(rewards))
            # print("statevalue "+str(state_values))
            # print("entropies "+str(entropies))
            returns = np.array([np.sum(rewards[i:] * discounts[:len(rewards)-i]) for i in range(len(rewards))])
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

            loss = 0
            for log_prob, value, R, entropy in zip(log_probs, state_values, returns, entropies):
                advantage = R - value.item()
                actor_loss = -(log_prob * advantage) - self.beta * entropy
                critic_loss = F.mse_loss(torch.tensor([R], dtype=torch.float32).to(self.device), value)
                loss += actor_loss + critic_loss
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.time()

    # def update(self, rewards, log_probs, state_values, entropies):
    #     rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

    #     discounts = torch.tensor([self.gamma ** i for i in range(len(rewards)+1)], device=self.device)
    #     returns = torch.zeros_like(rewards)
    #     for t in reversed(range(len(rewards)-1)):
    #         returns[t] = rewards[t] + discounts[t+1] * returns[t+1] * (1-int(t == len(rewards)-1))

    #     state_values = torch.tensor(state_values).to(self.device)
    #     advantages = returns - state_values.squeeze()

    #     actor_loss = -torch.stack(log_probs) * advantages.detach()
    #     actor_loss -= self.beta * torch.stack(entropies)
    #     critic_loss = F.mse_loss(state_values.squeeze(), returns.detach())
    #     loss = actor_loss.mean() + critic_loss

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
    #     self.optimizer.step()

    #     return loss.item()

    def evaluate(self, env, n_eval_episodes=30, max_episode_length=500):
        total_rewards = []
        
        for _ in range(n_eval_episodes):
            state, info = env.reset()
            episode_rewards = []
            done = False
            truncated = False
            iteration = 0
            
            while not done and not truncated and iteration < max_episode_length:
                action = self(state)
                state, reward, done, truncated, info = env.step(action)
                episode_rewards.append(reward)
                iteration += 1
            
            total_rewards.append(sum(episode_rewards))
        
        mean_return = np.mean(total_rewards)
        return mean_return
    # def evaluate(self, env, n_eval_episodes=30, max_episode_length=500):
    #     total_rewards = []
        
    #     for _ in range(n_eval_episodes):
    #         state, info = env.reset()
    #         episode_rewards = []
    #         done = False
    #         truncated = False
    #         iteration = 0
            
    #         while not done and not truncated and iteration < max_episode_length:
    #             action, log_prob, entropy = self.select_action(state)
    #             state, reward, done, truncated, info = env.step(action)
    #             episode_rewards.append(reward)
    #             iteration += 1
            
    #         total_rewards.append(sum(episode_rewards))
        
    #     mean_return = np.mean(total_rewards)
    #     return mean_return