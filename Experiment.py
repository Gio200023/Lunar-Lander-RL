#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import argparse
import sys

from Q_ActorCritic_LunarLender import actorcritic 
from Helper import LearningCurvePlot, smooth

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

def average_over_repetitions(n_repetitions, n_timesteps, learning_rate, gamma, smoothing_window=None, eval_interval=500, render_mode="",entropy=0.01):

    returns_over_repetitions = []
    now = time.time()
    
    for rep in range(n_repetitions): 
        
        returns, timesteps = actorcritic(n_timesteps, learning_rate, gamma,eval_interval,render_mode,beta=entropy)
        returns_over_repetitions.append(returns)
        print("Done nr: ", rep)

    print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    learning_curve = np.mean(np.array(returns_over_repetitions),axis=0) # average over repetitions
    if smoothing_window is not None: 
        learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve, timesteps  

def experiment():
    ####### Settings
    n_repetitions = 1
    smoothing_window = 9 # Must be an odd number. Use 'None' to switch smoothing off!
    render_mode= "rgb_array"
        
    n_timesteps = 300001 # Set one extra timestep to ensure evaluation at start and end
    eval_interval = 500
    
    # gammas = [0.1,0.99]
    # learning_rates = [0.01,0.001]
    # entropies = [0.01,0.9]
    gammas = [0.99]
    learning_rates = [0.01]
    entropies = [0.9]
    
    Plot = LearningCurvePlot(title = "Actor-Critic")
    Plot.set_ylim(-600, 200) 
    for learning_rate in learning_rates:
        for gamma in gammas:
            for entropy in entropies:
                print("Training with settings:")
                print(f"learning_rate = {learning_rate}")
                print(f"gamma = {gamma}")
                print(f"entropy = {entropy}")
                learning_curve, timesteps = average_over_repetitions(n_repetitions=n_repetitions, n_timesteps=n_timesteps,
                                                                        learning_rate=learning_rate, gamma=gamma, smoothing_window=smoothing_window, 
                                                                        eval_interval=eval_interval,render_mode=render_mode, entropy=entropy)
                
                Plot.add_curve(timesteps,learning_curve,label=("lr:"+str(learning_rate)+"gamma:"+str(gamma)+"entropy:"+str(entropy)))
            
    Plot.save('new_plots/lol.png')

if __name__ == '__main__':
    # args = get_args()
    experiment()