##################### Izlemesiz Kod #####################

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from Helper import LearningCurvePlot, smooth

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

def evaluate(model, env, timestep):
     while step < timestep:

        total_rewards = []
        state, _ = env.reset()
        done = False
        truncated = 0
        while not done and not truncated:

                probs, value = model(state)

                next_state, reward, done, truncated, _ = env.step(action.item())
                cumulative_reward += reward
                step += 1  # Increment step counter

                if next_state is not None:
                    next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                    log_prob = dist.log_prob(action)
                    log_probs.append(log_prob)
                    values.append(value)
                    rewards.append(torch.tensor([reward], dtype=torch.float))
                    masks.append(torch.tensor([1-done], dtype=torch.float))
                    # entropies.append(entropy)
                    entropies += entropy
                    state = next_state
                
                if step > timesteps:
                    break    

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def train(timesteps, env, model, episodes, gamma=0.99, lr=0.001, entropy_beta=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    episode_rewards = []  # To store cumulative rewards for each episode

    # for episode in range(episodes):
    step = 0  # Step counter to enforce max steps per episode
    while step < timesteps:
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        log_probs = []
        values = []
        rewards = []
        masks = []
        done = False
        cumulative_reward = 0  # To store cumulative reward for this episode
        truncated = False
        # entropies = []  # To store the entropies for entropy regularization
        entropies = 0  # To store the entropies for entropy regularization

        while not done and not truncated: 
            # if episode % 10 == 0:  # Render every 10 episodes
            #     env.render()

            probs = model.actor(state)
            dist = Categorical(probs)
            action = dist.sample()
            entropy = dist.entropy()
            value = model.critic(state)

            next_state, reward, done, truncated, _ = env.step(action.item())
            cumulative_reward += reward
            step += 1  # Increment step counter

            if next_state is not None:
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                log_prob = dist.log_prob(action)
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.tensor([reward], dtype=torch.float))
                masks.append(torch.tensor([1-done], dtype=torch.float))
                # entropies.append(entropy)
                entropies += entropy
                state = next_state
            
            if step > timesteps:
                break


        # After episode ends
        episode_rewards.append(cumulative_reward)

        if next_state is not None:
            _, next_value = model(next_state)
            returns = compute_returns(next_value, rewards, masks)

            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns).detach()
            values = torch.cat(values)
            # entropies = torch.cat(entropies)

            advantage = returns - values
            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()
            entropy_loss = -entropies.mean()

            total_loss = actor_loss + critic_loss - entropy_beta * entropy_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # if episode % 10 == 0:
        #     print(f'Episode {episode}, Loss: {total_loss.item()}, Reward: {cumulative_reward}')

    env.close()
    return episode_rewards  # Return the rewards for all episodes

# Initialize and train
env = gym.make('LunarLander-v2')
model = ActorCritic(input_dim=8, output_dim=4)
rewards = train(20000, env, model, episodes=5000, entropy_beta=0.01, gamma=0.99)

Plot = LearningCurvePlot(title="Actor-Critic")
Plot.add_curve(range(len(rewards)), rewards, label="Reward")
Plot.save('actor_critic_rewards.png')

##################### Izlemesiz Kod #####################



############### TESTING OF THE ENVIRONMENT  #######################

# import gym

# # Initialize the environment
# env = gym.make('LunarLander-v2')
# state = env.reset()

# # Take a random action
# action = env.action_space.sample()
# result = env.step(action)

# # Print the result to see what it includes
# print("Output of env.step():", result)
# print("Length of output:", len(result))

##################################################################