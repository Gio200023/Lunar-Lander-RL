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
    def __init__(self, n_states, n_actions, learning_rate, gamma):
        super(Q_ActorCritic_Agent, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma  
        self.learning_rate = learning_rate
        self._current_iteration = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logprobs = []
        self.state_values = []
        self.rewards = []

        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(n_states, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )
        
        #initialize actor network
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0)
        
        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(n_states, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
         #initialize critic network
        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0)
        
        # self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        # self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        self.to(self.device)

    def select_action_(self,policy_dist):
        dist = distributions.Categorical(policy_dist)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def forward(self, state):
        torch.autograd.set_detect_anomaly(True)

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        value = self.critic(state)
        logits= self.actor(state)
        policy_dist = F.softmax(logits, dim=1)
        
        return value, policy_dist

    def update_both(self,state, rewards, log_probs, values, entropy):
        
        values = torch.tensor(values,dtype=torch.float32).to(self.device).detach()
        
        #  Prepare Qvals without inplace operations
        Qvals_list = []
        q_val, _ = self.forward(state)
        q_val = q_val.detach()  # Ensure no gradients are propagated back from future computations
        for t in reversed(range(len(rewards))):
            q_val = rewards[t] + self.gamma * q_val
            Qvals_list.append(q_val)

        # Qvals = torch.zeros_like(values)
        # q_val, _ = self.forward(state)
        # for t in reversed(range(len(rewards))):
        #     q_val = rewards[t] + self.gamma * q_val
        #     Qvals[t] = q_val

        Qvals = torch.stack(Qvals_list[::-1]).to(self.device).detach()

        log_probs = torch.stack(log_probs)
        advantages = (Qvals - values).detach()
        advantages = advantages - advantages.mean() 
        
        actor_loss = (-log_probs *advantages).mean() + (0.01 *  entropy)
        critic_loss = 0.5 * advantages.pow(2).mean() + (0.01 *  entropy)
        ac_loss = actor_loss + critic_loss 
        
        self.optimizer.zero_grad()
        ac_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
    def update_bootstrap_only(self, state, rewards, log_probs, values, entropy):
        
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        
        # Qvals = torch.zeros_like(values)
        # q_val, _ = self.forward(state)
        # for t in reversed(range(len(rewards))):
        #     q_val = rewards[t] + self.gamma * q_val
        #     Qvals[t] = q_val
        
         # Prepare Qvals without inplace operations
        Qvals_list = []
        q_val, _ = self.forward(state)
        q_val = q_val.detach()  # Ensure no gradients are propagated back from future computations
        for t in reversed(range(len(rewards))):
            q_val = rewards[t] + self.gamma * q_val
            Qvals_list.append(q_val)

        Qvals = torch.stack(Qvals_list[::-1]).to(self.device)

        
        log_probs = torch.stack(log_probs)
        advantages = Qvals - values  # Only bootstrapping, no baseline subtraction
        
        actor_loss = (-log_probs * advantages).mean() + (0.01 *  entropy)
        critic_loss = 0.5 * advantages.pow(2).mean() + (0.01 *  entropy)
        ac_loss = actor_loss + critic_loss 

        self.optimizer.zero_grad()
        ac_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

    def update_baseline_only(self, state, rewards, log_probs, values, entropy):
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        
        baseline = values.mean()
        advantages = values - baseline  # Only using baseline to calculate advantages
        
        log_probs = torch.stack(log_probs)
        actor_loss = (-log_probs * advantages).mean() + (0.01 *  entropy)
        critic_loss = 0.5 * advantages.pow(2).mean() + (0.01 *  entropy)
        ac_loss = actor_loss + critic_loss 

        self.optimizer.zero_grad()
        ac_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
    def update_td(self, state, next_state, action, reward, entropy):
        current_value, probs = self(state)
        next_value, _ = self(next_state)
        
        td_target = reward + 0.99 * next_value.detach()
        td_error = td_target - current_value

        critic_loss = td_error.pow(2).mean()  
        
        log_prob = torch.log(probs.squeeze(0)[action].clamp(min=1e-6))
        app = -log_prob * td_error.detach()
        actor_loss =  app - 0.01 * entropy  
        loss = actor_loss + critic_loss

        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        # self.optimizer.step()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()


    def evaluate(self, env, n_eval_episodes=10, max_episode_length=200):
        total_rewards = []
        
        for _ in range(n_eval_episodes):
            state, info = env.reset()
            episode_rewards = []
            done = False
            truncated = False
            iteration = 0
            
            while not done and not truncated and iteration < max_episode_length:
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                policy_dist = self.actor(state)
                action = torch.argmax(policy_dist).item()
                state, reward, done, truncated, info = env.step(action)
                episode_rewards.append(reward)
                iteration += 1
            
            total_rewards.append(sum(episode_rewards))
        
        mean_return = np.mean(total_rewards)
        return mean_return