#################### Actor-Critic w/ Bootstrapping #####################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
from Helper import LearningCurvePlot, smooth

class ActorCritic_bootstrap(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic_bootstrap, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

def update_policy_bootstrap(state, action, reward, next_state, done, model, optimizer):
    logits, current_value = model(state)
    _, next_value = model(next_state)

    td_target = reward + 0.99 * next_value.detach() * (1 - int(done))
    td_error = td_target - current_value

    critic_loss = td_error.pow(2)
    probs = F.softmax(logits, dim=-1) + 1e-8
    actor_loss = -torch.log(probs.squeeze(0)[action]) * td_error.detach()
    loss = actor_loss + critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_and_evaluate_bootstrap(Plot):
    env = gym.make("LunarLander-v2")
    model = ActorCritic_bootstrap(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    rewards = []
    evaluation_rewards = []
    eval_time = []
    for episode in range(5000):
        state, _ = env.reset()
        state = torch.FloatTensor(state)
        episode_reward = 0
        done = False
        while not done:
            logits, _ = model(state)
            # logits = torch.clamp(logits, -10, 10)
            probs = F.softmax(logits, dim=-1)

            if np.random.rand() < 0.1:  # Epsilon-greedy for exploration
                action = env.action_space.sample()
            else:
                action = probs.multinomial(1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = torch.FloatTensor(next_state)

            update_policy_bootstrap(state, action, reward, next_state, done, model, optimizer)

            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)
        if (episode + 1) % 10 == 0:
            total_eval_reward = np.mean(rewards[-10:])
            eval_time.append(episode)
            evaluation_rewards.append(total_eval_reward)
            # evaluation_rewards.append(total_eval_reward)
            print(f"Episode: {episode + 1}, Average Reward (last 10): {total_eval_reward}")

    env.close()
    del model
    evaluation_rewards = smooth(evaluation_rewards,9)   
    Plot.add_curve(eval_time, evaluation_rewards, label="boot")

# #################### Actor-Critic w/ Bootstrapping #####################


# #################### Actor-Critic w/ Baseline Subtraction #####################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
from Helper import LearningCurvePlot

class ActorCritic_baseline(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic_baseline, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

def update_policy_baseline(state, action, reward, model, optimizer):
    logits, current_value = model(state)

    # Remove next_value and bootstrapping from the TD target
    # TD target becomes just the reward, because there's no bootstrapping
    td_target = torch.tensor([reward])  # Ensure this is a tensor for operations

    # Calculate advantage using only current reward and current value
    advantage = td_target - current_value

    critic_loss = advantage.pow(2)  # MSE Loss for the critic
    probs = F.softmax(logits, dim=-1) + 1e-8
    actor_loss = -torch.log(probs.squeeze(0)[action]) * advantage.detach()  # Actor loss using advantage

    loss = actor_loss + critic_loss  # Total loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_and_evaluate_baseline(Plot):
    env = gym.make("LunarLander-v2")
    model = ActorCritic_baseline(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    rewards = []
    evaluation_rewards = []
    eval_time = []

    for episode in range(5000):
        state, _ = env.reset()
        state = torch.FloatTensor(state)
        episode_reward = 0
        done = False
        while not done:
            logits, _ = model(state)
            probs = F.softmax(logits, dim=-1)

            if np.random.rand() < 0.1:  # Epsilon-greedy for exploration
                action = env.action_space.sample()
            else:
                action = probs.multinomial(1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = torch.FloatTensor(next_state)

            update_policy_baseline(state, action, reward, model, optimizer)  # Notice the updated function call

            state = next_state
            episode_reward += reward
        rewards.append(episode_reward)
        if (episode + 1) % 10 == 0:
            total_eval_reward = np.mean(rewards[-10:])
            eval_time.append(episode)
            evaluation_rewards.append(total_eval_reward)
            print(f"Episode: {episode + 1}, Average Reward (last 10): {total_eval_reward}")

    env.close()
    del model
    evaluation_rewards = smooth(evaluation_rewards,9)
    Plot.add_curve(eval_time, evaluation_rewards, label="base")
    
#################### Actor-Critic w/ Baseline Subtraction #####################

##################### Actor-Critic w/ Both #####################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
from Helper import LearningCurvePlot

class ActorCritic_both(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic_both, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value
    
def update_policy(state, action, reward, next_state, done, model, optimizer):
    logits, current_value = model(state)
    _, next_value = model(next_state)

    # Using the value function as a baseline
    td_target = reward + 0.99 * next_value.detach() * (1 - int(done))
    advantage = td_target - current_value  # Total advantage calculation

    critic_loss = advantage.pow(2)  # MSE Loss for the critic
    probs = F.softmax(logits, dim=-1) + 1e-8
    actor_loss = -torch.log(probs.squeeze(0)[action]) * advantage.detach()  # Actor loss using advantage
    loss = actor_loss + critic_loss  # Total loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_and_evaluate_both(Plot):
    env = gym.make("LunarLander-v2")
    model = ActorCritic_both(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    rewards = []
    evaluation_rewards = []
    eval_time = []

    for episode in range(5000):
        state, _ = env.reset()
        state = torch.FloatTensor(state)
        episode_reward = 0
        done = False
        while not done:
            logits, _ = model(state)
            probs = F.softmax(logits, dim=-1)

            if np.random.rand() < 0.1:  # Epsilon-greedy for exploration
                action = env.action_space.sample()
            else:
                action = probs.multinomial(1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = torch.FloatTensor(next_state)

            update_policy(state, action, reward, next_state, done, model, optimizer)

            state = next_state
            episode_reward += reward
        rewards.append(episode_reward)
        if (episode + 1) % 10 == 0:
            total_eval_reward = np.mean(rewards[-10:])
            evaluation_rewards.append(total_eval_reward)
            eval_time.append(episode)
            print(f"Episode: {episode + 1}, Average Reward (last 10): {total_eval_reward}")

    env.close()
    del model
    evaluation_rewards = smooth(evaluation_rewards,9)
    Plot.add_curve(eval_time, evaluation_rewards, label="both")

from multiprocessing import Process, Manager

if __name__ == '__main__':
    Plot = LearningCurvePlot(title = "ACTOR CRITIC BOTH")
    train_and_evaluate_both(Plot)
    train_and_evaluate_baseline(Plot)
    train_and_evaluate_bootstrap(Plot)

    Plot.save("barbie-actor-critic-cpu-hope_last.png")
##################### Actor-Critic w/ Both #####################

#################### Entropy Regularization #####################

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import gym
# import numpy as np
# from Helper import LearningCurvePlot

# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorCritic, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.actor = nn.Linear(128, action_dim)
#         self.critic = nn.Linear(128, 1)
#         self.optimizer = optim.Adam(self.parameters(), lr=0.001)

#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         logits = self.actor(x)
#         value = self.critic(x)
#         return logits, value

# def update_policy(state, action, reward, next_state, done, model, optimizer):
#     logits, current_value = model(state)
#     _, next_value = model(next_state)

#     td_target = reward + 0.99 * next_value.detach() * (1 - int(done))
#     td_error = td_target - current_value

#     critic_loss = td_error.pow(2)

#     probs = F.softmax(logits, dim=-1)
#     log_probs = F.log_softmax(logits, dim=-1)
#     actor_loss = -(log_probs.squeeze(0)[action] * td_error.detach()).mean()  # Compute actor loss

#     # Entropy Regularization
#     entropy = -(probs * log_probs).sum(dim=-1).mean()  # Compute entropy
#     loss = actor_loss + critic_loss - 0.01 * entropy  # Add entropy term to the loss

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# def train_and_evaluate():
#     env = gym.make("LunarLander-v2")
#     model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     rewards = []
#     evaluation_rewards = []

#     for episode in range(5000):
#         state, _ = env.reset()
#         state = torch.FloatTensor(state)
#         episode_reward = 0
#         done = False
#         while not done:
#             logits, _ = model(state)
#             probs = F.softmax(logits, dim=-1)

#             if np.random.rand() < 0.1:  # Epsilon-greedy for exploration
#                 action = env.action_space.sample()
#             else:
#                 action = probs.multinomial(1).item()

#             next_state, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated
#             next_state = torch.FloatTensor(next_state)

#             update_policy(state, action, reward, next_state, done, model, optimizer)

#             state = next_state
#             episode_reward += reward

#         rewards.append(episode_reward)
#         if (episode + 1) % 3 == 0:
#             total_eval_reward = np.mean(rewards[-10:])
#             evaluation_rewards.append(total_eval_reward)
#             print(f"Episode: {episode + 1}, Average Reward (last 10): {total_eval_reward}")

#     env.close()

#     # Plotting
#     Plot = LearningCurvePlot(title="Actor-Critic")
#     Plot.add_curve(range(len(evaluation_rewards)), evaluation_rewards, label="Reward")
#     Plot.save('actor_critic_entropy.png')

# if __name__ == '__main__':
#     train_and_evaluate()

#################### Entropy Regularization #####################