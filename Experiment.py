import numpy as np
import time
import argparse
import sys
import multiprocessing
from multiprocessing import Process, Manager

from Actor_critic import train_and_evaluate_entropy, train_and_evaluate_baseline,train_and_evaluate_bootstrap,train_and_evaluate_both
from Helper import LearningCurvePlot, smooth
from REINFORCE_LunardLander import reinforce


def average_over_repetitions(n_timesteps, learning_rate, gamma, smoothing_window=None, eval_interval=500, render_mode="",return_dict=None):
    
    now = time.time()

    returns, timesteps, max_min = reinforce(n_timesteps, learning_rate, gamma,eval_interval,render_mode,beta=0.99)

    print(returns)
    print('Running REINFORCE takes {} minutes'.format((time.time()-now)/60))

    # if smoothing_window is not None: 
    #     returns = smooth(returns,smoothing_window) # additional smoothing
        
    return_dict.append([returns, timesteps,"reinforce"])

def experiment():
    ####### Settings
    smoothing_window = 9 # Must be an odd number. Use 'None' to switch smoothing off!
    render_mode= "rgb_array"
        
    n_timesteps = 500000 # Set one extra timestep to ensure evaluation at start and end
    eval_interval = 2000
    
    # Plot = LearningCurvePlot(title = "ACTOR CRITIC vs REINFORCE")
    Plot = LearningCurvePlot(title = "ACTOR CRITIC")
    
    learning_rate = 0.001
    gamma = 0.9
    
    params = []
    manager = Manager()
    return_dict = manager.list()
    procs = []
    
    now = time.time()
    
    ## UNCOMMENT TRAIN TO RUN

    # proc = Process(target=average_over_repetitions,args=(n_timesteps,learning_rate, gamma, smoothing_window, eval_interval,render_mode, return_dict))
    # procs.append(proc)
    # proc.start()
    
    proc1 = Process(target=train_and_evaluate_both,args=(return_dict,n_timesteps,eval_interval))
    procs.append(proc1)
    proc1.start()
    
    # proc2 = Process(target=train_and_evaluate_baseline,args=(return_dict,n_timesteps,eval_interval))
    # procs.append(proc2)
    # proc2.start()
    
    # proc3 = Process(target=train_and_evaluate_bootstrap,args=(return_dict,n_timesteps,eval_interval))
    # procs.append(proc3)
    # proc3.start()
    
    proc4 = Process(target=train_and_evaluate_entropy,args=(return_dict,n_timesteps,eval_interval))
    procs.append(proc4)
    proc4.start()

    for proc in procs:
        proc.join()
    
    finish = time.time() - now
    
    for _ in range(len(return_dict)):
        Plot.add_curve(return_dict[_][1],return_dict[_][0],label=(f"{return_dict[_][2]}"))
        
    image_name = "entropy_egreedy_high.png"
    print(f"Training took: {finish} seconds, image name: {image_name} ")
    Plot.save(image_name)

if __name__ == '__main__':
    # args = get_args()
    experiment()