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
from ReplayBuffer import ReplayMemory
from ActorCriticLunarLander import Actor, Critic

class DQNAgent(nn.Module):
    """
        DQN Agent class, with e-greedy policy and experience replay buffer
    
    Raises:
        KeyError: Provide an epsilon
        KeyError: Provide a temperature

    Returns:
        int: best action according to the policy
    """
    def __init__(self, n_states, n_actions, learning_rate, gamma, epsilon=0.001, epsilon_decay=0.995, epsilon_min=0.01, temp=0.05, temp_decay = 0.995, temp_min = 0.01,target_update=50):
        super(DQNAgent, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.replay_buffer = ReplayMemory(10000000)
        self.gamma = gamma  
        self.temp = temp
        self.temp_decay = temp_decay
        self.temp_min = temp_min
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.Q_sa = np.zeros((n_states,n_actions))
        self._current_iteration = 0
        self.target_update = target_update
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Network
        self.layer1 = nn.Linear(self.n_states, 64)  
        # self.layer2 = nn.Linear(64, 64)  
        self.layer3 = nn.Linear(64, self.n_actions) 
        
        # Target Network
        self.target_layer1 = nn.Linear(self.n_states, 64)
        # self.target_layer2 = nn.Linear(64, 64)
        self.target_layer3 = nn.Linear(64, self.n_actions)
        
        #Hypertuning
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Initialization of the networks weights
        init.xavier_uniform_(self.layer1.weight)
        self.layer1.bias.data.fill_(0.0)
        # init.xavier_uniform_(self.layer2.weight)
        # self.layer2.bias.data.fill_(0.0)
        init.xavier_uniform_(self.layer3.weight)
        self.layer3.bias.data.fill_(0.0)

        init.xavier_uniform_(self.target_layer1.weight)
        self.target_layer1.bias.data.fill_(0.0)
        # init.xavier_uniform_(self.target_layer2.weight)
        # self.target_layer2.bias.data.fill_(0.0)
        init.xavier_uniform_(self.target_layer3.weight)
        self.target_layer3.bias.data.fill_(0.0)

        self.update_target_network()  # Initialize target network to be the same as the main network
        
        # Ensure the target network is not updated during backpropagation
        for param in self.target_layer1.parameters():
            param.requires_grad = False
        # for param in self.target_layer2.parameters():
        #     param.requires_grad = False
        for param in self.target_layer3.parameters():
            param.requires_grad = False
        
        self.to(self.device)
        
    def select_action(self, s, policy='egreedy'):
        state = torch.from_numpy(np.array(s)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self(state)

        if policy == 'softmax':
            if self.temp is None:
                raise KeyError("Provide a temperature")
            probabilities = F.softmax(q_values / self.temp, dim=-1).cpu().numpy().squeeze()
            action = np.random.choice(self.n_actions, p=probabilities)
            # Temperature decay
            self.temp = max(self.temp_min, self.temp * self.temp_decay)
            return action

        elif policy == 'egreedy':
            if np.random.rand() < self.epsilon:
                action = np.random.randint(self.n_actions)
            else:
                action = q_values.argmax().item()
            # Epsilon decay
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            return action

        else:
            return q_values.argmax().item()

    def update_target_network(self):
        # Helper method to update the target network
        self.target_layer1.load_state_dict(self.layer1.state_dict())
        # self.target_layer2.load_state_dict(self.layer2.state_dict())
        self.target_layer3.load_state_dict(self.layer3.state_dict())
    
    def forward(self, x, target=False):
        # x = F.relu(self.layer1(x))
        # # x = self.dropout1(x)  # Apply dropout after activation
        # x = F.relu(self.layer2(x))
        # return self.layer3(x)c

        if target:
            x = F.relu(self.target_layer1(x))
            # x = F.relu(self.target_layer2(x))
            x = self.target_layer3(x)
        else:
            x = F.relu(self.layer1(x))
            # x = F.relu(self.layer2(x))
            x = self.layer3(x)
        return x

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

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
        
    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=500, epsilon = 0.05,temp = 0.05):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s , info= eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = self.select_action(s=s, policy='greedy')
                observation, reward, terminated, truncated, info = eval_env.step(a)
                R_ep += reward
                if terminated:
                    break
                else:
                    s = observation
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return

class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, hidden_size, lr_actor, lr_critic, gamma):
        from ActorCriticLunarLander import Actor, Critic 
        self.actor = Actor(state_dim, action_dim, hidden_size)
        self.critic = Critic(state_dim, hidden_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = torch.softmax(self.actor(state), dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()

    def update(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64).view(-1, 1)
        reward = torch.tensor(reward, dtype=torch.float32).view(-1, 1)
        done = torch.tensor(done, dtype=torch.float32).view(-1, 1)

        # Compute TD target
        with torch.no_grad():
            target_value = reward + (1 - done) * self.gamma * self.critic(next_state)

        # Compute advantage
        critic_value = self.critic(state)
        advantage = target_value - critic_value

        # Actor Loss
        action_probs = torch.softmax(self.actor(state), dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(action.squeeze(-1))
        actor_loss = -(log_probs * advantage.detach()).mean()

        # Critic Loss
        critic_loss = nn.MSELoss()(critic_value, target_value)

        # Update networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()