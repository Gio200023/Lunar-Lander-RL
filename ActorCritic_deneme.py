import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, lr):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

def update_policy(state, action, reward, next_state, done, model, optimizer):
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

def train_and_evaluate(gamma, lr, episodes):
    env = gym.make("LunarLander-v2")
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n, lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    rewards = []
    avg_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state)
        episode_reward = 0
        done = False
        while not done:
            logits, _ = model(state)
            probs = F.softmax(logits, dim=-1)
            action = probs.multinomial(1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = torch.FloatTensor(next_state)

            update_policy(state, action, reward, next_state, done, model, optimizer)
            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            avg_rewards.append(avg_reward)
            print(f"Episode: {episode + 1}, Average Reward (last 10): {avg_reward}")

    env.close()
    return np.array(avg_rewards)

# Configurations for the experiments
configs = [
    (0.99, 0.001, "Gamma 0.99, LR 0.001"),
    (1.0, 0.01, "Gamma 1.0, LR 0.01"),
    (0.99, 0.01, "Gamma 0.99, LR 0.01"),
    (1.0, 0.001, "Gamma 1.0, LR 0.001")
]

plt.figure(figsize=(10, 8))
for gamma, lr, label in configs:
    rewards = train_and_evaluate(gamma, lr, 3000)
    episodes = np.linspace(0, 3000, len(rewards))
    plt.plot(episodes, rewards, label=label)
    plt.fill_between(episodes, rewards - np.std(rewards), rewards + np.std(rewards), alpha=0.1)

plt.title("Actor-Critic Training with Various Hyperparameters")
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.legend()
plt.show()
