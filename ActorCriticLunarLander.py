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
        log_probs = []
        values = []
        rewards = []
        masks = []
        done = False
        cumulative_reward = 0
        while not done and not truncated:

                probs, value = model(state)

                next_state, reward, done, truncated, _ = env.step(torch.action.item())
                cumulative_reward += reward
                step += 1  # Increment step counter

                if next_state is not None:
                    next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                    log_prob = torch.dist.log_prob(torch.action)
                    log_probs.append(log_prob)
                    values.append(value)
                    rewards.append(torch.tensor([reward], dtype=torch.float))
                    masks.append(torch.tensor([1-done], dtype=torch.float))
                    # entropies.append(entropy)
                    entropies += torch.entropy
                    state = next_state
                
                if step > timestep:
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


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import gym
# import numpy as np

# # Define Actor Network
# class Actor(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_size=64):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, output_dim)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return self.softmax(x)


# # Define Critic Network
# class Critic(nn.Module):
#     def __init__(self, input_dim, hidden_size=64):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)


# # Define Actor-Critic Agent
# class ActorCriticAgent(nn.Module):
#     def __init__(self, input_dim, action_dim, hidden_size=64, entropy_coef=0.01, lr=0.001):
#         super(ActorCriticAgent, self).__init__()
#         self.actor = Actor(input_dim, action_dim, hidden_size)
#         self.critic = Critic(input_dim, hidden_size)
#         self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
#         self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
#         self.entropy_coef = entropy_coef

#     def select_action(self, state):
#         state = torch.FloatTensor(state).unsqueeze(0)
#         action_probs = self.actor(state)
#         action_dist = torch.distributions.Categorical(action_probs)
#         action = action_dist.sample()
#         return action.item(), action_probs.squeeze(0)[action.item()]

#     def update(self, state, action, reward, next_state, done):
#         state = torch.FloatTensor(state)
#         next_state = torch.FloatTensor(next_state)
#         action_probs = self.actor(state)
#         action_dist = torch.distributions.Categorical(action_probs)
#         log_probs = action_dist.log_prob(action)
        
#         # Compute TD Target
#         with torch.no_grad():
#             td_target = reward + (1 - done) * 0.99 * self.critic(next_state)
#             advantage = td_target - self.critic(state)

#         # Update Critic
#         critic_loss = advantage.pow(2).mean()
#         self.optimizer_critic.zero_grad()
#         critic_loss.backward()
#         self.optimizer_critic.step()

#         # Update Actor
#         actor_loss = -(log_probs * advantage.detach() + self.entropy_coef * action_dist.entropy())
#         self.optimizer_actor.zero_grad()
#         actor_loss.mean().backward()
#         self.optimizer_actor.step()


# # Hyperparameters
# env = gym.make('LunarLander-v2')
# input_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
# hidden_size = 128
# lr = 0.001
# entropy_coef = 0.01

# # Initialize Agent
# agent = ActorCriticAgent(input_dim, action_dim, hidden_size, entropy_coef, lr)

# # Training Loop
# num_episodes = 1000
# for i_episode in range(num_episodes):
#     state = env.reset()[0]  # Extract the array part of the state
#     total_reward = 0
#     done = False
#     while not done:
#         state = np.array(state)  # Convert state to numpy array
#         action, _ = agent.select_action(state)
#         step_result = env.step(action)
#         next_state, reward, done, _ = step_result[:4]  # Extract the first four elements
#         next_state = next_state[0]  # Extract the array part of the next state
#         agent.update(state, action, reward, next_state, done)
#         state = next_state
#         total_reward += reward
#     print(f"Episode: {i_episode+1}, Total Reward: {total_reward}")





# # PREVIOUS CODE - DID NOT WORK

# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # import gym

# # num_iterations = 1000 
# # num_eval_episodes = 10 
# # eval_interval = 1000  

# # initial_collect_steps = 100  
# # collect_steps_per_iteration = 1
# # replay_buffer_max_length = 10000

# # batch_size = 64  
# # log_interval = 200

# # learning_rate = 1e-3  
# # gamma = 0.99
# # epsilon = 0.05
# # temp = 0.05

# # class ActorCriticAgent:
# #     def __init__(self, state_dim, action_dim, hidden_size, lr_actor, lr_critic, gamma):
# #         self.actor = Actor(state_dim, action_dim, hidden_size)
# #         self.critic = Critic(state_dim, hidden_size)
# #         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
# #         self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
# #         self.gamma = gamma

# #     def select_action(self, state):
# #         state = torch.tensor(state, dtype=torch.float32)
# #         action_probs = torch.softmax(self.actor(state), dim=-1)
# #         action_dist = torch.distributions.Categorical(action_probs)
# #         action = action_dist.sample()
# #         return action.item()

# #     def update(self, state, action, reward, next_state, done):
# #         state = torch.tensor(state, dtype=torch.float32)
# #         next_state = torch.tensor(next_state, dtype=torch.float32)
# #         action = torch.tensor(action, dtype=torch.int64).view(-1, 1)
# #         reward = torch.tensor(reward, dtype=torch.float32).view(-1, 1)
# #         done = torch.tensor(done, dtype=torch.float32).view(-1, 1)

# #         # Compute TD target
# #         with torch.no_grad():
# #             target_value = reward + (1 - done) * self.gamma * self.critic(next_state)

# #         # Compute advantage
# #         critic_value = self.critic(state)
# #         advantage = target_value - critic_value

# #         # Actor Loss
# #         action_probs = torch.softmax(self.actor(state), dim=-1)
# #         action_dist = torch.distributions.Categorical(action_probs)
# #         log_probs = action_dist.log_prob(action.squeeze(-1))
# #         actor_loss = -(log_probs * advantage.detach()).mean()

# #         # Critic Loss
# #         critic_loss = nn.MSELoss()(critic_value, target_value)

# #         # Update networks
# #         self.actor_optimizer.zero_grad()
# #         actor_loss.backward()
# #         self.actor_optimizer.step()

# #         self.critic_optimizer.zero_grad()
# #         critic_loss.backward()
# #         self.critic_optimizer.step()

# # class Actor(nn.Module):
# #     def __init__(self, state_dim, action_dim, hidden_size):
# #         super(Actor, self).__init__()
# #         self.fc1 = nn.Linear(state_dim, hidden_size)
# #         self.fc2 = nn.Linear(hidden_size, action_dim)

# #     def forward(self, state):
# #         x = torch.relu(self.fc1(state))
# #         x = self.fc2(x)
# #         return x

# # class Critic(nn.Module):
# #     def __init__(self, state_dim, hidden_size):
# #         super(Critic, self).__init__()
# #         self.fc1 = nn.Linear(state_dim, hidden_size)
# #         self.fc2 = nn.Linear(hidden_size, 1)

# #     def forward(self, state):
# #         x = torch.relu(self.fc1(state))
# #         x = self.fc2(x)
# #         return x

# # env = gym.make('LunarLander-v2')
# # # env = gym.make("LunarLander-v2",render_mode=render_mode, continuous = False,gravity = -10.0,enable_wind = False)
# # state_dim = env.observation_space.shape[0]
# # action_dim = env.action_space.n
# # hidden_size = 128
# # lr_actor = 0.001
# # lr_critic = 0.001
# # gamma = 0.99
# # agent = ActorCriticAgent(state_dim, action_dim, hidden_size, lr_actor, lr_critic, gamma)
# # num_episodes = 1000

# # for episode in range(num_episodes):
# #     state = env.reset()
# #     total_reward = 0
# #     done = False
# #     while not done:
# #         action = agent.select_action(state)
# #         next_state, reward, done, _ = env.step(action)
# #         agent.update(state, action, reward, next_state, done)
# #         state = next_state
# #         total_reward += reward
# #     print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# # env.close()
