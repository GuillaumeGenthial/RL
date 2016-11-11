from collections import OrderedDict
import numpy as np
import random
import sys
import time
from plotting import PlotManager
import copy
import pickle

def sarsa(file_name, nstates, nactions, niter=1000, alpha=0.01, gamma=0.95, Q=None):
    """
    Performs Sarsa learning over a file of samples
    No exploration strategy is needed, as the files provide the exploration itself
    Args
        file_name (string): file no read in (s, a, r, sp)
        nstates (int): number of states
        nactions (int): number of actions
        niter (int): number of maximum iteration over the file
        alpha (float): learning rate for sarsa, between 0 and 1
        gamma (float): discount factor, between 0 and 1
        tolerance (float): level at which if we don't improve the difference between 2 consecutives
            values of Q, we stop iteration
    Returns
        Q (np array dim=(nactions, nstates)): Q value arrays for each state actions
    """
    t0 = time.time()
    if Q is None:
        Q = np.zeros((nstates, nactions))
    else:
        Q = Q
    i, d = 0, 1
    while d > 1e-1 and i < niter:
        i += 1
        Q_tm1 = copy.deepcopy(Q)
        # are we starting a new game?
        jump = True
        njump = 0
        # iterate over exploration trajectories from file
        with open(file_name) as f:
            s_tm1, s_t, a_tm1, r_tm1 = 1, 1, 1, 0
            for index, line in enumerate(f):
                if index != 0:
                    # we suppose that lines are well ordered
                    s, a, r, sp = map(int, line.strip().split(","))
                    # if we are continuing a game, update, else start new game
                    if s_t != s:
                        njump += 1
                        jump = True
                    else:
                        jump = False
                    if not jump:
                        Q[s_tm1-1, a_tm1-1] = Q[s_tm1-1, a_tm1-1] + alpha * (r_tm1 + gamma * Q[s-1, a-1] - Q[s_tm1-1, a_tm1-1])
                    s_tm1, s_t, a_tm1, r_tm1 = s, sp, a, r
        t = time.time()
        if (t-t0) > 60:
            save(Q, "result/tmp_q_large.pkl")
            t0 = t
        d = np.sum(np.square(Q - Q_tm1))
        print "Info : iter {}, d = {}, njump = {}, total = {}".format(i, d, njump, index)
    return Q

def sarsa_lambda(file_name, nstates, nactions, niter=5000, alpha=0.1, gamma=0.95, lbda=0.9,  tolerance=0.1, auto_save=False, file_out="tmp.policy"):
    Q = np.zeros((nstates, nactions))
    N = np.zeros((nstates, nactions))
    i, d, d_ref = 0, 0, None
    while d > tolerance and i < niter or i == 0:
        i += 1
        Q_tm1 = copy.deepcopy(Q)
        # are we starting a new game?
        jump , njump = True, 0
        # iterate over exploration trajectories from file
        with open(file_name) as f:
            s_tm1, s_t, a_tm1, r_tm1 = 1, 1, 1, 0
            for index, line in enumerate(f):
                if index != 0:
                    # we suppose that lines are well ordered
                    s, a, r, sp = map(int, line.strip().split(","))
                    N[s-1, a-1] += 1
                    # if we are continuing a game, update, else start new game
                    if s_t != s:
                        njump += 1
                        jump = True
                    else:
                        jump = False
                    if not jump:
                        delta = r_tm1 + gamma * Q[s-1, a-1] - Q[s_tm1-1, a_tm1-1]
                        for s in range(nstates):
                            for a in range(nactions):
                                Q[s-1, a-1] = Q[s-1, a-1] + alpha * delta * N[s-1, a-1]
                                N[s-1, a-1] = gamma * lbda * N[s-1, a-1]
                    s_tm1, s_t, a_tm1, r_tm1 = s, sp, a, r
        d = np.sum(np.square(Q - Q_tm1))
        alpha, d_ref = update_learning_rate(alpha, d, d_ref)
        print "Info : iter {}, d = {}, njump = {}, total = {}, alpha = {}".format(i, d, njump, index, alpha)
        if auto_save == True:
            P = optimal_policy_from_Q(Q)
            write_policy_to_file(P, file_out)
    plt_mgr.close()
    return Q


def optimal_policy_from_Q(Q):
    """
    Given a Q function, returns the policy
    Args
        Q (np array dim=(nstates, nactions)) : Q(s, a) = Q value for tuple state action
    Returns
        P (np array dim = (nstates)) : P[s] = best action to take
    """
    return np.argmax(Q, axis=-1)


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

def save(Q, file):
    """
    saves Q into a file with pickle
    """
    with open(file, "w") as f:
        pickle.dump(Q, f)

def load(file):
    """
    Loads Q from file
    """
    with open(file) as f:
        return pickle.load(f)

def plot(Q):
    # plotting the Q function just to see in 3d
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = [], [], []
    for s in range(1757600):
        for a in range(7):
            p = np.random.rand()
            if p < 0.01 and Q[s,a] != 0:
                x += [s]
                y += [a]
                z += [Q[s, a]]
    ax.scatter(x, y, z)
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')
    ax.set_zlabel('Q value')
    plt.show()

def plot2(Q, a=0, p=0.5, r=1757600, o=0):
    """
    ploting for a given action
    """
    import matplotlib.pyplot as plt
    x, y = [], []
    for s in range(r):
        s = s + o
        _p = np.random.rand()
        if _p < p and Q[s,a] != 0:
            x += [s]
            y += [Q[s, a]]
    plt.plot(x, y)
    plt.show()

def get_neighbors(s, a, r, nstates=1757600, nactions=8):
    """
    Getting neighbors in a range r
    """
    s_n = [s-r, s, s+r]
    a_n = [a-r, a, a+r]
    neighbors = []
    for sp in s_n:
        for ap in a_n:
            if sp >=0 and sp<nstates and ap >= 0 and ap<nactions and (sp, ap) != (s, a):
                neighbors += [(sp, ap)]

    return neighbors

def approximate_Q(Q, nstates=1757600, nactions=8):
    """
    Computes Q value for unobserved pairs from neighbors
    """
    for s in range(nstates):
        for a in range(nactions):
            if Q[s,a] == 0:
                found = False
                i = 1
                while not found:
                    q, n = 0, 0
                    neighbors = get_neighbors(s, a, i)
                    for (sn, an) in neighbors:
                        if Q[sn-1, an-1] != 0:
                            n += 1
                            q += Q[sn-1, an-1]
                    if n != 0:
                        Q[s-1, a-1] = float(q)/float(n)
                        found = True
                    else:
                        i += 1
    return Q



file_name = "data/large.csv"
gamma = 0.99
# Q = sarsa(file_name, 1757600, 8, niter=1000, gamma=gamma)
# save(Q, "result/large_q.pkl")
Q = load("result/large_q.pkl")
print np.shape(Q)
Q = approximate_Q(Q)
# Q = sarsa(file_name, 1757600, 8, niter=1, gamma=gamma, Q=Q)
# save(Q, "result/large_q.pkl")
# plot2(Q, r=100000, o=300000)
P = optimal_policy_from_Q(Q)
write_policy_to_file(P, "large.policy")
