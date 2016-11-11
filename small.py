from collections import OrderedDict
import numpy as np
import random
import sys
import time
from plotting import PlotManager
import copy

def estimate_TR_from_file(file_name, nstates, nactions):
    """
    Estimates T and R from a file where each line is
    state_t, action, reward, state_{t+1}
    Assumes that the reward is deterministic
    Reads each sample and updates a count of transition and rewards.
    Args
        file_name (string): file to read samples in (s, a, r, sp)
        nstates (int): number of states for the problem
        nactions (int): number of actions
    Returns
        T (np array dim = (nstates, nactions, nstates)) : T(s, a, s') = P (s' | s, a)
        probabilities of transition
        R (np array dim = (nstates, nactions)) : rewards for each tuple
    """
    # keep records of rewards
    R = np.zeros((nstates, nactions))
    # keep record of transitions
    N = np.zeros((nstates, nactions, nstates))
    # iterate over file of samples
    with open(file_name) as f:
        for index, line in enumerate(f):
            if index != 0:
                s, a, r, sp = map(int, line.strip().split(","))
                R[s-1, a-1] = r
                N[s-1, a-1, sp-1] += 1

    # T(s, a, s') = P (s' | s, a)
    T = N / np.sum(N, axis=-1, keepdims=True)
    return T, R

def value_iteration(T, R, gamma=0.95, max_steps=5000):
    """
    Performs value iteration over a model defined by T and R
    Stops when improvement is lower than 10^{-1} or max_steps has been reached
    Args
        T (np array dim = (nstates, nactions, nstates)) : T(s, a, s') = P (s' | s, a)
        probabilities of transition
        R (np array dim = (nstates, nactions)) : rewards for each tuple
        gamma (float): must be between 0 and 1, discount factor
        max_steps (int): maximum steps for value iteration
    Returns
        U (np array dim = (nstates)): utility of each state
    """
    nstates = T.shape[0]
    U_k = np.zeros(nstates)
    criteria = True
    step = 0
    while criteria:
        step += 1
        # initialize new U array
        U = np.zeros_like(U_k)
        U_k = U_k[np.newaxis, np.newaxis, :]
        for s in xrange(nstates):
            U = np.max(R + gamma*np.sum(T*U_k, axis=-1), axis=-1)
        # controll progress and display it
        d = np.mean(np.square(U - U_k))
        sys.stdout.write("\rInfo : d = {}".format(d))
        sys.stdout.flush()
        criteria = (d > 10**(-15)) and step < max_steps
        U_k = U
    print "\nInfo : EU = %.4f" % np.mean(U_k)

    return U_k

def optimal_policy(T, R, U, gamma=0.95):
    """
    Computes optimal policy from T, R and U
    Args
        T (np array dim = (nstates, nactions, nstates)) : T(s, a, s') = P (s' | s, a)
        probabilities of transition
        R (np array dim = (nstates, nactions)) : rewards for each tuple
        gamma (float): must be between 0 and 1, discount factor
        U (np array dim = (nstates)): utility of each state
    Returns
        P (np array dim = (nstates)) : P[s] = best action to take
    """
    U = U[np.newaxis, np.newaxis, :]
    return np.argmax(R + gamma*np.sum(T*U, axis=-1), axis=-1)

def write_policy_to_file(P, file_name):
    """
    Given a policy, writes it into a file where line i contains id of best action in state i
    Args
        P (np array dim = (nstates)) : Q[s] = best action to take
        file_name (string): where to write the result
    """
    with open(file_name, "w") as f:
        for i, p in enumerate(P):
            f.write(str(p+1))
            if i != P.size - 1:
                f.write("\n")

def maximum_likelihood_RL(file_name, nstates, nactions, gamma=0.95, dyna=False, prop_update=0.1):
    """
    Performs maximum likelihood over samples from a file
    Args
        file_name (string): file to read samples in
        nstates (int): number of states for the problem
        nactions (int): number of actions
        gamma (float): must be between 0 and 1, discount factor
        dyna (bool): if True, will update only a subset of the Q function at each step
        prop_update (float): must be between 0 and 1, proportion of Q(s, a) to update
    Returns
        Q (np array dim=(nstates, nactions)) : Q(s, a) = Q value for tuple state action
    """
    # N global count, ro rewards, Q_t Q value function
    N = np.zeros((nstates, nactions, nstates))
    ro = np.zeros((nstates, nactions))
    Q_t = np.zeros((nstates, nactions))
    ntot = nstates * nactions
    ntot_update = int(prop_update * ntot)

    # loop over the file
    with open(file_name) as f:
        for index, line in enumerate(f):
            if index != 0:
                # update count and reward
                s, a, r, sp = map(int, line.strip().split(","))
                N[s-1, a-1, sp-1] += 1
                ro[s-1, a-1] += r
                # computes R and T
                R = ro / (np.sum(N, axis=-1) + 1e-6)
                T = N / (np.sum(N, axis=-1, keepdims=True) + 1e-6)
                # if dynamic, update only a proportion of the Q values
                if dyna:
                    q = Q_t[s-1, a-1]
                    # update Q value for current state action
                    Q_t[s-1, a-1] = R[s-1, a-1] + gamma * np.sum(T[s-1, a-1, :] * np.max(Q_t, axis=-1), axis=-1)
                    d = np.square(q - Q_t[s-1, a-1])
                    # update Q value for a random fraction of state actions
                    indices = random.sample(range(1, ntot), ntot_update)
                    for i in indices:
                        s, a = i/nactions, i%nactions
                        q = Q_t[s-1, a-1]
                        Q_t[s-1, a-1] = R[s-1, a-1] + gamma * np.sum(T[s-1, a-1, :] * np.max(Q_t, axis=-1), axis=-1)
                        d += np.square(q - Q_t[s-1, a-1])
                    d = d / (ntot_update + 1)
                # else update all Q values
                else:
                    Q = R + gamma * np.sum(T * (np.max(Q_t, axis=-1)[np.newaxis, np.newaxis, :]), axis=-1)
                    d = np.mean(np.square(Q - Q_t))
                    Q_t = Q
                sys.stdout.write("\rInfo : d = {}".format(d))
                sys.stdout.flush()
    print ""
    return Q_t


def optimal_policy_from_Q(Q):
    """
    Given a Q function, returns the policy
    Args
        Q (np array dim=(nstates, nactions)) : Q(s, a) = Q value for tuple state action
    Returns
        P (np array dim = (nstates)) : P[s] = best action to take
    """
    return np.argmax(Q, axis=-1)

file_name = "data/small.csv"
gamma = 0.95


# 1. Model Based RL
# Dynamic Programming from estimation
T, R = estimate_TR_from_file(file_name, 100, 4)
U = value_iteration(T, R, gamma)
P = optimal_policy(T, R, U, gamma)
write_policy_to_file(P, "small_ref.policy")

# Maximum Likely hood method
Q = maximum_likelihood_RL(file_name, 100, 4, dyna=False)
P = optimal_policy_from_Q(Q)
write_policy_to_file(P, "small.policy")

# 2. Model free RL
# Sarsa (from files medium.py or large.py)
# Q = sarsa(file_name, 100, 4)
# P = optimal_policy_from_Q(Q)
# write_policy_to_file(P, "small.policy")



