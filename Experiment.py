#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import argparse
import sys

from REINFORCE_LunardLender import reinforce 
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

def average_over_repetitions(n_timesteps, learning_rate, gamma, smoothing_window=None, eval_interval=500, render_mode="",beta=0.01):
    
    now = time.time()
    
    returns, timesteps, max_min = reinforce(n_timesteps, learning_rate, gamma,eval_interval,render_mode,beta=beta)

    print(returns)
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    learning_curve = np.mean(np.array(returns),axis=0) # average over repetitions
    # if smoothing_window is not None: 
    #     learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return returns, timesteps, max_min

def experiment():
    ####### Settings
    smoothing_window = 9 # Must be an odd number. Use 'None' to switch smoothing off!
    render_mode= "rgb_array"
        
    n_timesteps = 300001 # Set one extra timestep to ensure evaluation at start and end
    eval_interval = 500
    
    learning_rates = [0.01,0.001]
    gammas = [0.1,0.99]
    betas = [0.1,0.9]
    # gammas = [0.99]
    # learning_rates = [0.001]
    # betas = [0.9]
    
    Plot = LearningCurvePlot(title = "REINFORCE")
    Plot.set_ylim(-300, 300)
    for learning_rate in learning_rates:
        for gamma in gammas:
            for beta in betas:
                print(f"Paramters set: \n lr: {learning_rate}\n gamma: {gamma} \n beta {beta}")
                learning_curve, timesteps, max_min = average_over_repetitions(n_timesteps=n_timesteps,learning_rate=learning_rate, 
                                                                     gamma=gamma, smoothing_window=smoothing_window, 
                                                                     eval_interval=eval_interval,render_mode=render_mode, beta=beta)
                

                Plot.add_fill_between(timesteps,[item[0] for item in max_min],[item[1] for item in max_min])
                Plot.add_curve(timesteps,learning_curve,label=("lr:"+str(learning_rate)+"gam:"+str(gamma)+"beta:"+str(beta)))
            
    Plot.save('new_plots/reinforce_lunar_lander_300k_multiple_param_big_net.png')

if __name__ == '__main__':
    # args = get_args()
    experiment()

