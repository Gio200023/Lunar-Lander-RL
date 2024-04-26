import torch
import torch.nn as nn
import torch.optim as optim
import gym

# num_iterations = 1000 
# num_eval_episodes = 10 
# eval_interval = 1000  

# initial_collect_steps = 100  
# collect_steps_per_iteration = 1
# replay_buffer_max_length = 10000

# batch_size = 64  
# log_interval = 200

# learning_rate = 1e-3  
# gamma = 0.99
# epsilon = 0.05
# temp = 0.05

class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, hidden_size, lr_actor, lr_critic, gamma):
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

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.fc2(x)
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.fc2(x)
        return x

env = gym.make('LunarLander-v2')
# env = gym.make("LunarLander-v2",render_mode=render_mode, continuous = False,gravity = -10.0,enable_wind = False)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_size = 128
lr_actor = 0.001
lr_critic = 0.001
gamma = 0.99
agent = ActorCriticAgent(state_dim, action_dim, hidden_size, lr_actor, lr_critic, gamma)
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

env.close()
