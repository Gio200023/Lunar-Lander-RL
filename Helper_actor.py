#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class LearningCurvePlot:

    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots(figsize=(15, 8))
        # self.fig,self.ax = plt.subplots()
        self.fig.tight_layout(rect=[0, 0, 0.7, 1])
        self.ax.set_xlabel('Timestep')
        self.ax.set_ylabel('Episode Return')      
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self,x,y,label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(x,y,label=label)
        else:
            self.ax.plot(x,y)

    def add_fill_between(self,x,y,label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        print("y ",y)
        mean = np.mean(y, axis=0)
        std = np.std(y, axis=0)
        print("mean: ", mean)
        print("std: ", std)

        # self.ax.fill_between(x, mean - std, mean + std, alpha=.5, linewidth=0, label=label)
        self.ax.plot(x,y,label=label)
    
    def set_ylim(self,lower,upper):
        self.ax.set_ylim([lower,upper])

    def add_hline(self,height,label):
        self.ax.axhline(height,ls='--',c='k',label=label)

    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
        # self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.fig.savefig(name, dpi=300, bbox_inches='tight')
        # self.fig.savefig(name,dpi=300)


def smooth(y, window, poly=2):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)

def softmax(x, temp):
    ''' Computes the softmax of vector x with temperature parameter 'temp' '''
    x = x / temp # scale by temperature
    z = x - max(x) # substract max to prevent overflow of softmax 
    return np.exp(z)/np.sum(np.exp(z)) # compute softmax

def argmax(x):
    ''' Own variant of np.argmax with random tie breaking '''
    try:
        return np.random.choice(np.where(x == np.max(x))[0])
    except:
        return np.argmax(x)

def linear_anneal(t,T,start,final,percentage):
    ''' Linear annealing scheduler
    t: current timestep
    T: total timesteps
    start: initial value
    final: value after percentage*T steps
    percentage: percentage of T after which annealing finishes
    ''' 
    final_from_T = int(percentage*T)
    if t > final_from_T:
        return final
    else:
        return final + (start - final) * (final_from_T - t)/final_from_T

if __name__ == '__main__':
    # Test Learning curve plot
    x = np.arange(100)
    y = 0.01*x + np.random.rand(100) - 0.4 # generate some learning curve y
    LCTest = LearningCurvePlot(title="Test Learning Curve")
    LCTest.add_curve(x,y,label='method 1')
    LCTest.add_curve(x,smooth(y,window=35),label='method 1 smoothed')
    # LCTest.add_fill_between(x,,label='fill')
    LCTest.save(name='learning_curve_test.png')