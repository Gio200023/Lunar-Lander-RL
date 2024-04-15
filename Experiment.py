#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import argparse
import sys

from LunarLander import PolicyGradient
from Helper import LearningCurvePlot, smooth
from Agent import DQNAgent

def average_over_repetitions(n_repetitions, n_timesteps, max_episode_length, use_replay_buffer, learning_rate, 
                                          gamma, policy, epsilon, epsilon_decay, epsilon_min, temp, temp_min, temp_decay, smoothing_window=None, eval_interval=500,batch_size=64,
                                          use_target_network = True):

    returns_over_repetitions = []
    now = time.time()
    
    for rep in range(n_repetitions): 
        
        returns, timesteps = PolicyGradient(n_timesteps, use_replay_buffer, learning_rate, gamma, policy, epsilon, temp,eval_interval,batch_size=batch_size, use_target_network = use_target_network)
        returns_over_repetitions.append(returns)
        print("Done nr: ", rep)

    print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    learning_curve = np.mean(np.array(returns_over_repetitions),axis=0) # average over repetitions
    if smoothing_window is not None: 
        learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve, timesteps  

def get_args():
    parser = argparse.ArgumentParser(description="Experiment settings")
    parser.add_argument('--no_er', action='store_true', help='Do not use replay buffer if flag is set')
    parser.add_argument('--no_tn', action='store_true', help='Do not use target network if flag is set')
    
    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        print("Argument parsing error: ", e.message)
        sys.exit(2)
    except SystemExit:
        # This can be triggered if unknown arguments are provided.
        print("Incorrect usage")
        print("Possible arguments are:")
        print("--no_er: remove the Experience Replay")
        print("--no_tn: remove the Target Network")
        sys.exit(2)
    return args

def experiment(use_replay_buffer, use_target_network):
    ####### Settings
    n_repetitions = 20
    smoothing_window = 9 # Must be an odd number. Use 'None' to switch smoothing off!
    use_replay_buffer = not use_replay_buffer
    use_target_network = not use_target_network

    print("repl buf: ",use_replay_buffer)
    print("use_tar: ",use_target_network)
        
    n_timesteps = 50001 # Set one extra timestep to ensure evaluation at start and end
    eval_interval = 500
    max_episode_length = 500
    gamma = 0.99
    batch_size = 64
    # if use_replay_buffer:
    #     batch_size = 32
    # else:
    #     batch_size = 1
    
    policies = ['egreedy', 'softmax'] 
    epsilon = 0.001
    epsilon_min = 0.05
    epsilon_decay =0.995
    temp = 0.1
    temp_min = 0.01
    temp_decay = 0.995
    # Back-up & update
    learning_rates = [0.01,0.001,0.1]
    
    Plot = LearningCurvePlot(title = "DQN-TN-ER")
    Plot.set_ylim(0, 500) 
    for learning_rate in learning_rates:
        for policy in policies:
            learning_curve, timesteps = average_over_repetitions(n_repetitions=n_repetitions, n_timesteps=n_timesteps, max_episode_length=max_episode_length, use_replay_buffer = use_replay_buffer, 
                                                                 learning_rate=learning_rate, gamma=gamma, policy=policy, epsilon=epsilon, epsilon_decay=epsilon_decay , 
                                                                 epsilon_min=epsilon_min, temp=temp, temp_min=temp_min, temp_decay=temp_decay, smoothing_window=smoothing_window, 
                                                                 eval_interval=eval_interval,batch_size=batch_size, use_target_network = use_target_network)
            
            Plot.add_curve(timesteps,learning_curve,label=(str(learning_rate)+","+str(policy)))
            
    Plot.save('dqn_no_tn_no_er.png')

if __name__ == '__main__':
    args = get_args()
    experiment(args.no_er, args.no_tn)

