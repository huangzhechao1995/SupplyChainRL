import itertools
#import sys
#sys.path.append("/Users/jonahadler/Desktop/code/SupplyChainRL/")
import os
os.chdir('C:\\Users\\alesc\\Documents\\Scuola\\MIT\\6.883 Modelling with ML\\\SupplyChainRL-master')
import utils
import framework
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
"""Linear QL agent"""


DEBUG = False

GAMMA = 0.5  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 10
NUM_EPOCHS = 1200
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 1e-2  # learning rate for training
BIGM=1e6

K = 2
Kpr = 1
Kst = np.array([1, 1, 1]).reshape(K+1)
Kpe = 3
Ktr = np.array([np.nan, 1, 1]).reshape(K+1)
CWarehouse = np.array([150, 50, 50]).reshape(K+1)
CTruck = np.array([np.nan, 70, 70]).reshape(K+1)
Price = 10
dmax = 20
action_for_factory = [0, 50, 100]
action_for_facilities = [0, 15, 30]
NUM_ACTIONS = len(action_for_factory) * len(action_for_facilities)**K

#K = 2
#Kpr = 5
#Kst = np.array([0, 0, 0]).reshape(K+1)
#Kpe = 5
#Ktr = np.array([np.nan, 0, 0]).reshape(K+1)
#CWarehouse = np.array([20, 20, 20]).reshape(K+1)
#CTruck = np.array([np.nan, 10, 10]).reshape(K+1)
#Price = 10
#dmax = 2
#action_for_factory = [0, 3, 6,10,15]
#action_for_facilities = [0, 1, 2,8,10]
#NUM_ACTIONS = len(action_for_factory) * len(action_for_facilities)**K

#K = 2
potential_actions = [[x[0]]+list(x[1]) for x in list(itertools.product(
    action_for_factory, itertools.product(action_for_facilities, repeat=K)))]

def tuple2index(L):
    if type(L) is np.ndarray:
        L = list(L)
    return potential_actions.index(L)
  


def index2tuple(x):
    return np.array(potential_actions[x])


def epsilon_greedy(state, theta, epsilon):
    """Returns an action selected by an epsilon-greedy exploration policy

    Args:
        state_vector (np.ndarray): extracted vector representation
        theta (np.ndarray): current weight matrix
        epsilon (float): the probability of choosing a random command

    Returns:
        (int, int): the indices describing the action/object to take
    """
    state_vector = utils.extract_state_feature_vector(
           state)
    explore = np.random.random_sample() < epsilon

    if explore:
        #must account for what we are allowed to do!
        
        #print('explore')
        avail = available_action_indices(state[0])
        #print(sum(avail))
        avail_idx = np.argwhere(avail == 1).reshape(-1)
        #print(len(avail_idx))
        avail_choice = random.choice(avail_idx)
        action_arr = index2tuple(avail_choice)
    else:
        #print('best')
        avail = available_action_indices(state[0])
        #print(sum(avail))
        avail_idx = np.argwhere(avail == 1).reshape(-1)
        
        q_function=theta @ state_vector
        best_idx=0
        best_q=-100000
        for idx in range(NUM_ACTIONS):
            if idx in avail_idx:
                if q_function[idx]>best_q:
                    best_q= q_function[idx]
                    best_idx=idx
        
        action_arr = index2tuple(best_idx)
        #options = q_func[state_1, state_2, :, :]
        #action_index, object_index = np.unravel_index(
        #    options.argmax(), options.shape)

    return action_arr


def available_action_indices(stock):
  return(np.where(np.array(potential_actions)[:, 1:].sum(axis=1) <= stock[0], 1, 0))

def availible_account_for(state):
    return (1-available_action_indices(state[0]))*-BIGM

def linear_q_learning(theta, current_state, action_arr,
                      reward, next_state, terminal):
    """Update theta for a given transition

    Args:
        theta (np.ndarray): current weight matrix
        current_state_vector (np.ndarray): vector representation of current state
        action_index (int): index of the current action
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_vector (np.ndarray): vector representation of next state
        terminal (bool): True if this epsiode is over

    Returns:
        None
    """
    current_state_vector = utils.extract_state_feature_vector(
           current_state)
    next_state_vector = utils.extract_state_feature_vector(
        next_state)

    #next_out_of_bounds = (1-available_action_indices(next_state[0]))*-BIGM
    best_next = ((theta @ next_state_vector)+availible_account_for(next_state)).max()
    this_q = (theta @ current_state_vector)[
        tuple2index(action_arr)]
    neg_grad = (reward+(1-terminal)*GAMMA*best_next-this_q) * \
        current_state_vector
    theta[tuple2index(action_arr)] = theta[tuple2index(
        action_arr)] + ALPHA * neg_grad
    return None


def run_episode(for_training):
    """ Runs one episode
    If for training, update Q function
    If for testing, computes and return cumulative discounted reward

    Args:
        for_training (bool): True if for training

    Returns:
        None
    """
    epsilon = TRAINING_EP if for_training else TESTING_EP
    epi_reward = 0
    game = framework.NewsVendorGame(K, Kpr, Kst, Kpe, Ktr,
                                    CWarehouse, CTruck, Price, dmax)
    #load in initial state
    (current_state, terminal) = (
        game.stock, game.d, game.old_d) , False

    t = 0



    while not terminal:
        # Choose next action and execute
        #current_state_vector = utils.extract_state_feature_vector(
         #   current_state)
         
         
        action_arr = epsilon_greedy(
            current_state, theta, epsilon)
        
        #print('stock \t',game.stock)
        #print('actions\t', action_arr)
        
        (next_state,reward,terminal) = game.step_game(action_arr)
        #print('demand \t', game.d)
        #print('ends \t',game.stock)
        #print('---------------------')

        if for_training:
            # update Q-function.
            #next_state_vector = utils.extract_state_feature_vector(
            #    next_state)
            
            
            linear_q_learning(theta, current_state, action_arr,
                               reward, next_state, terminal)
            
            
            
        if not for_training:
            epi_reward += reward*GAMMA**t
            t += 1

        current_state= next_state

    if not for_training:
        return epi_reward


def run_epoch():
    """Runs one epoch and returns reward averaged over test episodes"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(for_training=False))

    return np.mean(np.array(rewards))


def run():
    """Returns array of test reward per epoch for one run"""
    global theta
    theta = np.zeros([action_dim, state_dim])

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    for _ in pbar:
        single_run_epoch_rewards_test.append(run_epoch())
        pbar.set_description(
            "Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(
                np.mean(single_run_epoch_rewards_test),
                utils.ewma(single_run_epoch_rewards_test)))
    return single_run_epoch_rewards_test


if __name__ == '__main__':
    pass
'''
state_texts = utils.load_data('game.tsv')
dictionary = utils.bag_of_words(state_texts)
framework.load_game_data()
'''

state_dim = K*3+1
action_dim = NUM_ACTIONS

# set up the game
game = framework.NewsVendorGame(K, Kpr, Kst, Kpe, Ktr,
                      CWarehouse, CTruck, Price, dmax)



epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS

for _ in range(NUM_RUNS):
    epoch_rewards_test.append(run())

epoch_rewards_test = np.array(epoch_rewards_test)

x = np.arange(NUM_EPOCHS)
fig, axis = plt.subplots()
axis.plot(x, np.mean(epoch_rewards_test,
                        axis=0))  # plot reward per epoch averaged per run
axis.set_xlabel('Epochs')
axis.set_ylabel('reward')
axis.set_title(('Linear: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
