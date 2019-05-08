#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 20:35:34 2019

@author: zhelwang
"""

import numpy as np
import matplotlib.pyplot as plt
import framework

def sigma_Q_policy(stock, sigma, Q):
    '''
    stock:      array of int, stock level for each warehouse and  
    sigma:      array of int, threshold of replenishment level
    Q:          array of int, specified amount for replenishment
    
    return:
    assignment: array of int, amount to replenish for each factory  
    '''    
    assignment = np.zeros([len(stock)])
    assignment[1:] = np.where(stock[1:] < sigma[1:], Q[1:],0)
    assignment[0] = (stock[0] - np.sum(assignment[1:]) < sigma[0]) * Q[0]
    return assignment 


def run_episode(num_iter, sigma, Q):
    """ Runs one episode
    If for training, update Q function
    If for testing, computes and return cumulative discounted reward

    Args:
        for_training (bool): True if for training

    Returns:
        None
    """
    epi_reward = 0
    reward_vec = []
    game = framework.NewsVendorGame(K, Kpr, Kst, Kpe, Ktr,
                                    CWarehouse, CTruck, Price, dmax)
    #load in initial state
    (current_state, terminal) = (
        game.stock, game.d, game.old_d) , False

    t = 0

    while not num_iter <= t:

        action_arr = sigma_Q_policy(game.stock, sigma, Q)
        (next_state,reward,terminal) = game.step_game(action_arr)
        reward_vec.append(reward)
        epi_reward += reward*GAMMA**t
        t += 1

    return epi_reward, reward_vec

def grid_search_reward(num_iter, CWarehouse, num_obj):
    max_cap = np.min(CWarehouse)
    max_avg_profit = -100000
    max_epi = -100000
    best_index_avg = (1,1)
    best_index_epi = (1,1)
    
    for i in range(max_cap):
        for j in range(max_cap):
            sigma = i * np.ones(num_obj)
            Q = j * np.ones(num_obj) 
            epi_reward, reward_vec = run_episode(num_iter, sigma, Q)
            mean_reward = np.mean(reward_vec)
            
            if mean_reward  > max_avg_profit:
                max_avg_profit = mean_reward  
                best_index_avg = (i,j)
                
            if epi_reward > max_epi:
                max_epi = epi_reward
                best_index_epi = (i,j)
    
    return max_epi, max_avg_profit, best_index_avg, best_index_epi

def run_best_param(num_iter, best_index, num_obj):
    sigma = best_index[0] * np.ones(num_obj)
    Q = best_index[1] * np.ones(num_obj) 
    epi_reward, reward_vec = run_episode(num_iter, sigma, Q)
    
    print("Best Policy Results")
    print('epi reward:', round(epi_reward,3))
    print('avg reward:', round(np.mean(reward_vec),3))
    plt.scatter(range(num_iter), reward_vec)

if __name__ == '__main__':
    
    K = 2
    Kpr = 5
    Kst = np.array([0, 0, 0]).reshape(K+1)
    Kpe = 5
    Ktr = np.array([np.nan, 0, 0]).reshape(K+1)
    CWarehouse = np.array([20, 20, 20]).reshape(K+1)
    CTruck = np.array([np.nan, 10, 10]).reshape(K+1)
    Price = 10
    dmax = 2
    GAMMA = 0.5
    
    num_obj = len(CWarehouse)
    num_iter = 36
    max_epi, max_avg_profit, best_index_avg, best_index_epi = grid_search_reward(num_iter, CWarehouse, num_obj)
    
    print('max epi reward:', round(max_epi,3))
    print('max avg reward:', round(max_avg_profit,3))
    print('----------')
    run_best_param(num_iter, best_index_avg, num_obj)
