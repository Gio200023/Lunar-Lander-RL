#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import argparse
import sys
import multiprocessing
from multiprocessing import Process, Manager

from REINFORCE_LunardLender import reinforce 
from Helper import LearningCurvePlot, smooth

# def get_args():
#     parser = argparse.ArgumentParser(description="Experiment settings")
#     parser.add_argument('--no_er', action='store_true', help='Do not use replay buffer if flag is set')
#     parser.add_argument('--no_tn', action='store_true', help='Do not use target network if flag is set')
#     try:
#         args = parser.parse_args()
#     except argparse.ArgumentError as e:
#         print("Argument parsing error: ", e.message)
#         sys.exit(2)
#     except SystemExit:
#         # This can be triggered if unknown arguments are provided.
#         print("Incorrect usage")
#         print("Possible arguments are:")
#         print("--no_er: remove the Experience Replay")
#         print("--no_tn: remove the Target Network")
#         sys.exit(2)
#     return args

def average_over_repetitions(n_timesteps, learning_rate, gamma, smoothing_window=None, eval_interval=500, render_mode="",beta=0.01,return_dict=None):
    
    now = time.time()
    returns, timesteps, max_min = reinforce(n_timesteps, learning_rate, gamma,eval_interval,render_mode,beta=beta)

    print(returns)
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    # learning_curve = np.mean(np.array(returns),axis=0) # average over repetitions
    # if smoothing_window is not None: 
    #     returns = smooth(returns,smoothing_window) # additional smoothing
    return_dict.append([returns, timesteps, max_min])

def experiment():
    ####### Settings
    smoothing_window = 9 # Must be an odd number. Use 'None' to switch smoothing off!
    render_mode= "rgb_array"
        
    n_timesteps = 4001 # Set one extra timestep to ensure evaluation at start and end
    eval_interval = 1000
    
    Plot = LearningCurvePlot(title = "REINFORCE")
    Plot.set_ylim(-300, 300)
    
    # learning_rates = [0.01,0.001]
    # gammas = [0.1,0.99]
    # betas = [0.01,0.9]
    gammas = [0.99]
    learning_rates = [0.001]
    betas = [0.9]
    
    params = []
    
    manager = Manager()
    return_dict = manager.list()
    
    procs = []
    # with multiprocessing.Pool(processes=3) as pool:
    for learning_rate in learning_rates:
        for gamma in gammas:
            for beta in betas:
                print(f"Paramters set: \n lr: {learning_rate}\n gamma: {gamma} \n beta {beta}")
                proc = Process(target=average_over_repetitions,args=(n_timesteps,learning_rate, gamma, smoothing_window, eval_interval,render_mode, beta,return_dict))
                procs.append(proc)
                proc.start()
                
                params.append([learning_rate,gamma,beta])


    for proc in procs:
        proc.join()
    
    for _ in range(len(return_dict)):
        Plot.add_fill_between(return_dict[_][1],return_dict[_][0],label=("lr:"+str(params[_][0])+"gam:"+str(params[_][1])+"beta:"+str(params[_][2])))
        # Plot.add_curve(return_dict[_][1],return_dict[_][0],label=("lr:"+str(params[_][0])+"gam:"+str(params[_][1])+"beta:"+str(params[_][2])))
            
    Plot.save('new_plots/lol.png')

if __name__ == '__main__':
    # args = get_args()
    experiment()

