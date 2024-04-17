# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Practical for master course 'Reinforcement Learning',
# Leiden University, The Netherlands
# By Thomas Moerland
# """

import numpy as np
from Helper import softmax, argmax
import random
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
        
        self.policy_network = nn.Sequential(
            nn.Linear(self.n_states, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        self.to(self.device)
        
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probabilities = self.policy_network(state)
        action = torch.multinomial(probabilities, 1).item()
        return action, probabilities
    
    def update_policy(self, rewards, log_probs):

        rewards = np.array(rewards)
        # Calculate discount factors in a vectorized manner
        discounts = np.power(self.gamma, np.arange(len(rewards)))
        # Calculate returns using a vectorized approach
        returns = np.array([np.sum(rewards[i:] * discounts[:len(rewards)-i]) for i in range(len(rewards))])

        # Convert returns to a torch tensor for compatibility with torch operations
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
    
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()    
    
    def replay(self, batch_size, use_target_network = True):
        
        minibatch = random.sample(self.replay_buffer.memory, batch_size)

        # Convert to numpy arrays first for efficiency
        states_np = np.array([transition.state.squeeze() for transition in minibatch])
        next_states_np = np.array([transition.next_state.squeeze() for transition in minibatch])
        actions_np = np.array([transition.action for transition in minibatch])
        rewards_np = np.array([transition.reward for transition in minibatch])
        dones_np = np.array([transition.done for transition in minibatch])
        
        # Now convert to PyTorch tensors
        states = torch.from_numpy(states_np).float().to(self.device)
        next_states = torch.from_numpy(next_states_np).float().to(self.device)
        actions = torch.from_numpy(actions_np).long().to(self.device).unsqueeze(-1)  # Actions are usually of type long
        rewards = torch.from_numpy(rewards_np).float().to(self.device)
        dones = torch.from_numpy(dones_np).float().to(self.device)
        
        # Compute the target Q values
        current_q_values = self(states).gather(1, actions)
        # Target true if using target network, target false for not use it.
        next_q_values = self(next_states, target=use_target_network).detach().max(1)[0].unsqueeze(-1)
        targets = rewards.unsqueeze(-1) + (1 - dones.unsqueeze(-1)) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, targets)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update the target network
        if self._current_iteration % self.target_update == 0:
            self.update_target_network()
        
    def evaluate(self,eval_env,n_eval_episodes=30):
        total_rewards = []
        for _ in range(n_eval_episodes):
            state, info = eval_env.reset()
            episode_reward = 0
            done = False
            while not done:
                action, _ = self.select_action(state)
                state, reward, done, truncated, info = eval_env.step(action)
                episode_reward += reward
            total_rewards.append(episode_reward)
        mean_return = np.mean(total_rewards)
        return mean_return