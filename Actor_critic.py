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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.to(self.device)

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

def train_and_evaluate_bootstrap(return_dict=None,n_timestep=500000,eval_interval=2000):
    env = gym.make("LunarLander-v2")
    model = ActorCritic_bootstrap(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    rewards = []
    evaluation_time = []
    eval_rewards = []
    iteration = 0
    while iteration <= n_timestep:
        state, _ = env.reset()
        state = torch.FloatTensor(state).to(model.device)
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
            next_state = torch.FloatTensor(next_state).to(model.device)

            update_policy_bootstrap(state, action, reward, next_state, done, model, optimizer)

            state = next_state
            episode_reward += reward
            if (iteration +1) % eval_interval == 0:
                total_eval_reward = np.mean(rewards[-1])
                eval_rewards.append(total_eval_reward) 
                evaluation_time.append(iteration)
                print(f"(boot) timestep: {iteration + 1}, Average Reward (last 10): {total_eval_reward}")
            iteration+=1
            if iteration >= n_timestep:
                break    
        rewards.append(episode_reward)

    env.close()
    del model
    eval_rewards = smooth(eval_rewards,9)
    # Plot.add_curve(range(len(evaluation_rewards)), evaluation_rewards, label="both")
    return_dict.append([eval_rewards,evaluation_time,"boot"])

# #################### Actor-Critic w/ Bootstrapping #####################


# #################### Actor-Critic w/ Baseline Subtraction #####################

class ActorCritic_baseline(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic_baseline, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.to(self.device)

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
    td_target = torch.tensor([reward]).to(model.device)  # Ensure this is a tensor for operations

    # Calculate advantage using only current reward and current value
    advantage = td_target - current_value

    critic_loss = advantage.pow(2)  # MSE Loss for the critic
    probs = F.softmax(logits, dim=-1) + 1e-8
    actor_loss = -torch.log(probs.squeeze(0)[action]) * advantage.detach()  # Actor loss using advantage

    loss = actor_loss + critic_loss  # Total loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_and_evaluate_baseline(return_dict=None,n_timestep=500000,eval_interval=2000):
    env = gym.make("LunarLander-v2")
    model = ActorCritic_baseline(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    rewards = []
    evaluation_time = []
    eval_rewards = []
    iteration = 0
    while iteration <= n_timestep:
        state, _ = env.reset()
        state = torch.FloatTensor(state).to(model.device)
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
            next_state = torch.FloatTensor(next_state).to(model.device)

            update_policy_baseline(state, action, reward, model, optimizer)  # Notice the updated function call

            state = next_state
            episode_reward += reward
            if (iteration + 1) % eval_interval == 0:
                total_eval_reward = np.mean(rewards[-1])
                eval_rewards.append(total_eval_reward) 
                evaluation_time.append(iteration)
                print(f"(base) timestep: {iteration + 1}, Average Reward (last 10): {total_eval_reward}")
            iteration+=1
            if iteration >= n_timestep:
                break
        rewards.append(episode_reward)

    env.close()
    del model
    # Plot.add_curve(range(len(evaluation_rewards)), evaluation_rewards, label="both")
    eval_rewards = smooth(eval_rewards,9)
    return_dict.append([eval_rewards,evaluation_time,"base"])
    
#################### Actor-Critic w/ Baseline Subtraction #####################

##################### Actor-Critic w/ Both #####################

class ActorCritic_both(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic_both, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.to(self.device)
        
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
    advantage = td_target - current_value

    critic_loss = advantage.pow(2) 
    probs = F.softmax(logits, dim=-1) + 1e-8
    actor_loss = -torch.log(probs.squeeze(0)[action]) * advantage.detach()  
    loss = actor_loss + critic_loss 

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_and_evaluate_both(return_dict=None,n_timestep=500000,eval_interval=2000):
    env = gym.make("LunarLander-v2")
    model = ActorCritic_both(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    rewards = []
    evaluation_time = []
    eval_rewards = []
    iteration = 0
    while iteration <= n_timestep:
        state, _ = env.reset()
        state = torch.FloatTensor(state).to(model.device)
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
            next_state = torch.FloatTensor(next_state).to(model.device)

            update_policy(state, action, reward, next_state, done, model, optimizer)

            state = next_state
            episode_reward += reward
            iteration+=1
            if (iteration + 1) % eval_interval == 0:
                total_eval_reward = np.mean(rewards[-1])
                eval_rewards.append(total_eval_reward) 
                evaluation_time.append(iteration)
                print(f"(both) timestep: {iteration + 1}, Average Reward (last 10): {total_eval_reward}")
            if iteration >= n_timestep:
                break
        rewards.append(episode_reward)

    env.close()
    del model
    
    eval_rewards = smooth(eval_rewards,9)
    return_dict.append([eval_rewards,evaluation_time,"both"])
##################### Actor-Critic w/ Both #####################

#################### Entropy Regularization #####################
class ActorCritic_entropy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic_entropy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

def update_policy_entropy(state, action, reward, next_state, done, model, optimizer):
    logits, current_value = model(state)
    _, next_value = model(next_state)

    td_target = reward + 0.99 * next_value.detach() * (1 - int(done))
    td_error = td_target - current_value

    critic_loss = td_error.pow(2).mean()

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    actor_loss = -(log_probs.squeeze(0)[action] * td_error.detach()).mean()

    # Entropy Regularization
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    loss = actor_loss + critic_loss - 0.01 * entropy  

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_and_evaluate_entropy(return_dict=None,n_timestep=5000,eval_interval=2000):
    env = gym.make("LunarLander-v2")
    model = ActorCritic_entropy(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    rewards = []
    eval_rewards = []
    evaluation_time = []
    iteration = 0
    while iteration <= n_timestep:
        state, _ = env.reset()
        state = torch.FloatTensor(state).to(model.device)
        episode_reward = 0
        done = False
        while not done:
            logits, _ = model(state)
            probs = F.softmax(logits, dim=-1)

            if np.random.rand() < 0.5:  # Epsilon-greedy for exploration
                action = env.action_space.sample()
            else:
                action = probs.multinomial(1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = torch.FloatTensor(next_state).to(model.device)

            update_policy_entropy(state, action, reward, next_state, done, model, optimizer)

            state = next_state
            episode_reward += reward
            if (iteration + 1) % eval_interval == 0:
                total_eval_reward = np.mean(rewards[-1])
                eval_rewards.append(total_eval_reward) 
                evaluation_time.append(iteration)
                print(f"(entropy) timestep: {iteration + 1}, Average Reward (last 10): {total_eval_reward}")
            iteration+=1
            if iteration >= n_timestep:
                break
        rewards.append(episode_reward)

    env.close()
    del model
    # Plot.add_curve(range(len(evaluation_rewards)), evaluation_rewards, label="both")
    eval_rewards = smooth(eval_rewards,9)
    return_dict.append([eval_rewards,evaluation_time,"entropy"])

#################### Entropy Regularization #####################

from multiprocessing import Process, Manager

if __name__ == '__main__':
    params = []
    procs = []
    manager = Manager()
    return_dict = manager.list()
    Plot = LearningCurvePlot(title = "ACTOR CRITIC with Entropy")
    
    n_timestep = 5000
    
    # proc1 = Process(target=train_and_evaluate_both,args=(return_dict,n_timestep))
    # procs.append(proc1)
    # proc1.start()
    
    # proc2 = Process(target=train_and_evaluate_baseline,args=(return_dict,n_timestep))
    # procs.append(proc2)
    # proc2.start()
    
    # proc3 = Process(target=train_and_evaluate_bootstrap,args=(return_dict,n_timestep))
    # procs.append(proc3)
    # proc3.start()
    
    proc4 = Process(target=train_and_evaluate_entropy,args=(return_dict,n_timestep))
    procs.append(proc4)
    proc4.start()
    
    for proc in procs:
        proc.join()
        
    for _ in range(len(return_dict)):
        Plot.add_curve(return_dict[_][1],return_dict[_][0],label=(f"{return_dict[_][2]}"))
    Plot.save("actorcritic-gpu-entropy.png")