import numpy as np
import time
import argparse
import sys
import multiprocessing
from multiprocessing import Process, Manager

from Q_ActorCritic_LunarLender import actorcritic 
from for_cart import train_and_evaluate
from Helper import LearningCurvePlot, smooth

def average_over_repetitions(n_timesteps, learning_rate, gamma, smoothing_window=None, eval_interval=500, render_mode="",return_dict=None,update="both"):
    
    now = time.time()
    # returns, timesteps = actorcritic(n_timesteps, learning_rate, gamma,eval_interval,render_mode,update=update)
    
    returns, timesteps = train_and_evaluate(n_timesteps,update,eval_interval, learning_rate, gamma, 0.90)

    print(returns)
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    # learning_curve = np.mean(np.array(returns),axis=0) # average over repetitions
    if smoothing_window is not None: 
        returns = smooth(returns,smoothing_window) # additional smoothing
    return_dict.append([returns, timesteps])

def experiment():
    ####### Settings
    smoothing_window = 9 # Must be an odd number. Use 'None' to switch smoothing off!
    render_mode= "rgb_array"
        
    n_timesteps = 500001 # Set one extra timestep to ensure evaluation at start and end
    eval_interval = 2000
    
    Plot = LearningCurvePlot(title = "ACTOR CRITIC")
    Plot.set_ylim(-500, 500)
    
    learning_rates = [0.001]
    gammas = [0.9]
    updates = ["both","base","boot"]
    
    # single run
    # updates = ["td"]
    # gammas = [0.99]
    # learning_rates = [0.001]
    
    # td
    # updates = ["td"]
    # gammas = [0.99,0.1]
    # learning_rates = [0.001, 0.01]
    
    params = []
    manager = Manager()
    return_dict = manager.list()
    
    procs = []
    
    now = time.time()

    for learning_rate in learning_rates:
        for gamma in gammas:
            for update in updates:
                print(f"Parameters set: \n lr: {learning_rate}\n gamma: {gamma}")
                proc = Process(target=average_over_repetitions,args=(n_timesteps,learning_rate, gamma, smoothing_window, eval_interval,render_mode, return_dict,update))
                procs.append(proc)
                proc.start()
                
                params.append([learning_rate,gamma,update])

    for proc in procs:
        proc.join()
    
    finish = time.time() - now
    
    for _ in range(len(return_dict)):
        if len(updates) > 1:
            Plot.add_curve(return_dict[_][1],return_dict[_][0],label=(f"{params[_][2]},gam:{params[_][1]},lr:{params[_][0]}"))
        else:
            Plot.add_curve(return_dict[_][1],return_dict[_][0],label=("lr:"+str(params[_][0])+"gam:"+str(params[_][1])))
    
    image_name = "new_plots/try.png"
    print(f"Training took: {finish} seconds, image name: {image_name} ")
    Plot.save(image_name)

if __name__ == '__main__':
    # args = get_args()
    experiment()