import itertools
import sys
sys.path.append("/Users/jonahadler/Desktop/code/SupplyChainRL/")
import os
os.chdir("/Users/jonahadler/Desktop/code/SupplyChainRL/")
#import utils
import framework
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import utils

#"deep q"

DEBUG = False

GAMMA = 0.5  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 10
NUM_EPOCHS = 1200
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 1e-2  # learning rate for training
BIGM = 1e8

K = 2
Kpr = 0
Kst = np.array([0, 0, 0]).reshape(K+1)
Kpe = 10
Ktr = np.array([np.nan, 0, 0]).reshape(K+1)
CWarehouse = np.array([100, 50, 50]).reshape(K+1)
CTruck = np.array([np.nan, 50, 50]).reshape(K+1)
Price = 10
dmax = 10
action_for_factory = [0, 20, 80]
action_for_facilities = [0, 20, 50]
NUM_ACTIONS = len(action_for_factory) * len(action_for_facilities)**K

potential_actions = [[x[0]]+list(x[1]) for x in list(itertools.product(
    action_for_factory, itertools.product(action_for_facilities, repeat=K)))]


def tuple2index(L):
    if type(L) is np.ndarray:
        L = list(L)
    return potential_actions.index(L)


def index2tuple(x):
    return np.array(potential_actions[x])


def epsilon_greedy(state, epsilon):
    """Returns an action selected by an epsilon-greedy exploration policy

    Args:
        state_vector (torch.FloatTensor): extracted vector representation
        theta (np.ndarray): current weight matrix
        epsilon (float): the probability of choosing a random command

    Returns:
        (int, int): the indices describing the action/object to take
    """
    state_vector = torch.FloatTensor(utils.extract_state_feature_vector(
        state))
    rand = np.random.choice([True, False], p=[epsilon, 1 - epsilon])
    if rand:
        avail = available_action_indices(state[0])
        avail_idx = np.argwhere(avail == 1).reshape(-1)
        avail_choice = random.choice(avail_idx)
        action_arr = index2tuple(avail_choice)
    else:
        with torch.no_grad():
            q_values_action = model(state_vector)
        maxq_next_vector = (
            q_values_action + available_account_for(state))
        action_index = np.argmax(maxq_next_vector).item()
        action_arr = index2tuple(action_index)

    return action_arr


class DQN(nn.Module):
    """A simple deep Q network implementation.
    Computes Q values for each (action, object) tuple given an input state vector
    """

    def __init__(self, state_dim, action_dim, hidden_size=100):
        super(DQN, self).__init__()
        self.state_encoder = nn.Linear(state_dim, hidden_size)
        self.state2action = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        state = F.relu(self.state_encoder(x))
        return self.state2action(state)


def available_action_indices(stock):
  return(np.where(np.array(potential_actions)[:, 1:].sum(axis=1) <= stock[0], 1, 0))


def available_account_for(state):
    return torch.FloatTensor((1-available_action_indices(state[0]))*-BIGM)


def deep_q_learning(current_state, action_arr, reward,
                    next_state, terminal):
    """Updates the weights of the DQN for a given transition

    Args:
        current_state_vector (torch.FloatTensor): vector representation of current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_vector (torch.FloatTensor): vector representation of next state
        terminal (bool): True if this epsiode is over

    Returns:
        None
    """
    current_state_vector = torch.FloatTensor(
        utils.extract_state_feature_vector(current_state))
    next_state_vector = torch.FloatTensor(
        utils.extract_state_feature_vector(next_state))
        
    with torch.no_grad():
        q_values_action_next = model(next_state_vector)
    maxq_next = (q_values_action_next +
                 available_account_for(next_state)).max()
    ## We need to check feasibility here

    # TODO Your code here
    q_values_action = model(current_state_vector)
    q_value_cur_state = q_values_action[tuple2index(action_arr)]
    y = reward + (1-terminal)*(GAMMA * maxq_next)
    loss = 1/2 * (q_value_cur_state - y) ** 2

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10000)
    optimizer.step()


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
        game.stock, game.d, game.old_d), False

    t = 0

    while not terminal:
        # Choose next action and execute
        #current_state_vector = utils.extract_state_feature_vector(
         #   current_state)
        current_state_vector = torch.FloatTensor(
            utils.extract_state_feature_vector(current_state))
        

        action_arr = epsilon_greedy(
            current_state, epsilon)
        (next_state, reward, terminal) = game.step_game(action_arr)

        if for_training:
            # update Q-function.

          deep_q_learning(current_state, action_arr,
                            reward, next_state, terminal)

        if not for_training:
            epi_reward += reward*GAMMA**t
            t += 1

        current_state = next_state

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
    #global theta
    #theta = np.zeros([action_dim, state_dim])

    global model
    global optimizer
    model = DQN(state_dim, NUM_ACTIONS)
    #model.double()
    optimizer = optim.SGD(model.parameters(), lr=ALPHA)

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
