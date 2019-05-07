import sys
sys.path.append("/Users/jonahadler/Desktop/code/SupplyChainRL/")
import utils
import framework
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
"""Linear QL agent"""


DEBUG = False


GAMMA = 0.5  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 10
NUM_EPOCHS = 1200
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 0.001  # learning rate for training

ACTIONS = []
NUM_ACTIONS = len(ACTIONS)



def tuple2index(action_index, object_index):
    """Converts a tuple (a,b) to an index c"""
    return action_index * NUM_OBJECTS + object_index


def index2tuple(index):
    """Converts an index c to a tuple (a,b)"""
    return index // NUM_OBJECTS, index % NUM_OBJECTS


def epsilon_greedy(state_vector, theta, epsilon):
    """Returns an action selected by an epsilon-greedy exploration policy

    Args:
        state_vector (np.ndarray): extracted vector representation
        theta (np.ndarray): current weight matrix
        epsilon (float): the probability of choosing a random command

    Returns:
        (int, int): the indices describing the action/object to take
    """

    explore = np.random.random_sample() < epsilon

    if explore:
        action_index = np.random.randint(NUM_ACTIONS)
        object_index = np.random.randint(NUM_OBJECTS)
    else:

        (action_index, object_index) = index2tuple(
            np.argmax(theta @ state_vector))
        #options = q_func[state_1, state_2, :, :]
        #action_index, object_index = np.unravel_index(
        #    options.argmax(), options.shape)

    return (action_index, object_index)


def linear_q_learning(theta, current_state_vector, action_index, object_index,
                      reward, next_state_vector, terminal):
    """Update theta for a given transition

    Args:
        theta (np.ndarray): current weight matrix
        current_state_vector (np.ndarray): vector representation of current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_vector (np.ndarray): vector representation of next state
        terminal (bool): True if this epsiode is over

    Returns:
        None
    """

    best_next = (theta @ next_state_vector).max()
    this_q = (theta @ current_state_vector)[
        tuple2index(action_index, object_index)]
    neg_grad = (reward+(1-terminal)*GAMMA*best_next-this_q) * \
        current_state_vector
    theta[tuple2index(action_index, object_index)] = theta[tuple2index(
        action_index, object_index)] + ALPHA * neg_grad
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

    (current_room_desc, current_quest_desc, terminal) = framework.newGame()

    t = 0

    while not terminal:
        # Choose next action and execute
        current_state = current_room_desc + current_quest_desc
        current_state_vector = utils.extract_bow_feature_vector(
            current_state, dictionary)

        (action_index, object_index) = epsilon_greedy(
            current_state_vector, theta, epsilon)
        (next_room_desc, next_quest_desc, reward, terminal) = framework.step_game(
            current_room_desc, current_quest_desc, action_index, object_index)

        if for_training:
            # update Q-function.
            next_state = next_room_desc + next_quest_desc
            next_state_vector = utils.extract_bow_feature_vector(
                next_state, dictionary)
            linear_q_learning(theta, current_state_vector, action_index,
                              object_index, reward, next_state_vector, terminal)

        if not for_training:
            epi_reward += reward*GAMMA**t
            t += 1

        current_room_desc, current_quest_desc = next_room_desc, next_quest_desc

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

state_texts = utils.load_data('game.tsv')
dictionary = utils.bag_of_words(state_texts)
state_dim = len(dictionary)
action_dim = NUM_ACTIONS * NUM_OBJECTS

# set up the game
framework.load_game_data()

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
