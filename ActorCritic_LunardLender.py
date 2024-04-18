import gym
import time
import numpy as np
from Agent import ActorCritic_Agent
import sys

# PARAMETERS if not initialized from Experiment.py
num_iterations = 1000 
num_eval_episodes = 10 
eval_interval = 1000  

initial_collect_steps = 100  
collect_steps_per_iteration =   1
replay_buffer_max_length = 10000

batch_size = 64  
log_interval = 200

learning_rate = 1e-3  
gamma = 0.99
epsilon = 0.05
temp = 0.05

def reinforce(n_timesteps=num_iterations, use_replay_buffer=True, learning_rate=learning_rate, gamma=gamma, 
        policy="egreedy", epsilon=epsilon, temp=temp, eval_interval=eval_interval, batch_size=batch_size, use_target_network = True):
    
    env = gym.make("LunarLander-v2",render_mode="human", continuous = False,gravity = -10.0,enable_wind = False)
    env_eval = gym.make("LunarLander-v2")
    epsilon_decay = 0.995
    epsilon_min = 0.05
    
    dqn_agent_and_model = DQNAgent(n_states=8, 
                        n_actions=4, 
                        learning_rate=learning_rate, 
                        gamma=gamma,
                        epsilon=epsilon,
                        epsilon_decay=epsilon_decay,
                        epsilon_min=epsilon_min,
                        temp=temp)
    
    actor = ActorCritic_Agent("actor")
    critic = ActorCritic_Agent("critic")
                        
    observation, info = env.reset(seed=42) 

    eval_timesteps = []
    eval_returns = []

    iteration = 0
    while iteration <= n_timesteps:
        state, info = env.reset()

        terminated = False
        while not terminated:
            action = dqn_agent_and_model.select_action(state,policy=policy)
            print(action)
            observation, reward, terminated, truncated, info = env.step(action)
                
            state = observation            
            
            if iteration % eval_interval == 0:
                eval_timesteps.append(iteration)
                eval_returns.append(dqn_agent_and_model.evaluate(env_eval, n_eval_episodes=num_eval_episodes, epsilon = epsilon, temp = temp))
                print("step: ",iteration)

            iteration+=1
            dqn_agent_and_model._current_iteration=iteration

            if iteration >= n_timesteps:
                break
            if terminated:
                break

    dqn_agent_and_model.replay_buffer.clean() 
    env.close()
    
    return np.array(eval_returns), np.array(eval_timesteps) 


if __name__ == '__main__':
    reinforce()