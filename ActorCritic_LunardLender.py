import gym
import numpy as np
from Agent import ActorCritic_Agent

# PARAMETERS if not initialized from Experiment.py
num_iterations = 1000
num_eval_episodes = 10
eval_interval = 100

batch_size = 64
learning_rate = 1e-3
gamma = 0.99
_entropy = 0.01

def actorcritic(n_timesteps=num_iterations, learning_rate=learning_rate, gamma=gamma, 
              eval_interval=eval_interval, render_mode="human",_entropy=_entropy):
    
    env = gym.make("LunarLander-v2", render_mode=render_mode)
    env_eval = gym.make("LunarLander-v2")

    reinforceAgent = ActorCritic_Agent(n_states=8, n_actions=4, learning_rate=learning_rate,
                              gamma=gamma, _entropy=_entropy)

    eval_timesteps = []
    eval_returns = []

    iteration = 0
    while iteration < n_timesteps:
        state, info = env.reset(seed=42) 
        episode_rewards = []
        log_probs = []
        state_values = []
        entropies = []
        done = False
        
        while not done:
            action, log_prob, entropy = reinforceAgent.select_action(state)
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_rewards.append(reward)
            log_probs.append(log_prob)
            _, state_value = reinforceAgent(state)
            state_values.append(state_value)
            entropies.append(entropy)
            
            state = observation
            iteration += 1
            
            # Evaluation step
            if iteration % eval_interval == 0:
                    eval_timesteps.append(iteration)
                    eval_returns.append(reinforceAgent.evaluate(env_eval, n_eval_episodes=num_eval_episodes))
                    print(f"Step: {iteration}, Average Return: {np.mean(eval_returns[-1])}")
        
            if iteration >= n_timesteps:
                break
            if terminated:
                break
        
        # Update the policy and value networks
        reinforceAgent.update(episode_rewards, log_probs, state_values, entropies)
        
    env.close()
    env_eval.close()
    
    return np.array(eval_returns), np.array(eval_timesteps)

if __name__ == '__main__':
    returns, timesteps = actorcritic()
