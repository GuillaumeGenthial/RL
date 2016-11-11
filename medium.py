from collections import OrderedDict
import numpy as np
import random
import sys
import time
from plotting import PlotManager
import copy
import pickle

def sarsa(file_name, nstates, nactions, niter=5000, alpha=0.01, gamma=0.95, Q=None):
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
    if Q is None:
        Q = np.zeros((nstates, nactions))
    else:
        Q = Q
    i, d = 0, 1
    while d > 1e-1 and i < niter:
        i += 1
        Q_tm1 = copy.deepcopy(Q)
        jump = True
        njump = 0
        with open(file_name) as f:
            s_tm1, s_t, a_tm1, r_tm1 = 1, 1, 1, 0
            for index, line in enumerate(f):
                if index != 0:
                    # we suppose that lines are well ordered
                    s, a, r, sp = map(int, line.strip().split(","))
                    if s_t != s:
                        njump += 1
                        jump = True
                    else:
                        jump = False
                    if not jump:
                        Q[s_tm1-1, a_tm1-1] = Q[s_tm1-1, a_tm1-1] + alpha * (r_tm1 + gamma * Q[s-1, a-1] - Q[s_tm1-1, a_tm1-1])
                    s_tm1, s_t, a_tm1, r_tm1 = s, sp, a, r
        d = np.sum(np.square(Q - Q_tm1))
        print "Info : iter {}, d = {}, njump = {}, total = {}".format(i, d, njump, index)
    return Q

def get_neighbours(s, act, r=1):
    """
    Get neighbors from sates actions in a range r from pos and vel
    """
    pos = (s-1)%500
    vel = (s-1)/500
    pos_n = []
    vel_n = []
    a_n = []
    for m in range(r):
        pos_n += [pos - m, pos + m]
        vel_n += [vel - m, vel + m]
        a_n += [act -m, act + m]
    neighbours = []
    for p in pos_n:
        for v in vel_n:
            for a in a_n:
                if (p >= 0 and p < 500) and (v >=0 and v < 100) and (a >0 and a <8) and ((p,v, a) != (pos, vel, act)):
                    neighbours += [(1 + p + 500*v, a)]

    return neighbours

def approximate_Q(Q, nstates, nactions):
    """
    Perform local approximation from neighbors for unboserved state action tuples
    """
    for s in range(nstates):
        for a in range(nactions):
            if Q[s-1, a-1] == 0:
                found = False
                i = 1
                while not found:
                    neighbours = get_neighbours(s, a, i)
                    q = 0
                    n = 0
                    for (sn, an) in neighbours:
                        if Q[sn-1, an-1] != 0:
                            n += 1
                            q += Q[sn-1, an-1]
                    if n != 0:
                        Q[s-1, a-1] = float(q)/float(n)
                        found = True
                    else:
                        i += 1
    return Q


def optimal_policy_from_Q(Q):
    return np.argmax(Q, axis=-1)

def write_policy_to_file(P, file_name):
    with open(file_name, "w") as f:
        for i, p in enumerate(P):
            f.write(str(p+1))
            if i != P.size - 1:
                f.write("\n")

def save(Q, file):
    with open(file, "w") as f:
        pickle.dump(Q, f)

def load(file):
    with open(file) as f:
        return pickle.load(f)

def plot(Q):
    # plotting the Q function just to see
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = [], [], []
    for s in range(50000):
        for a in range(7):
            p = np.random.rand()
            if p < 0.1 and Q[s,a] != 0:
                x += [s]
                y += [a]
                z += [Q[s, a]]
    ax.scatter(x, y, z)
    ax.set_xlabel('States')
    ax.set_ylabel('Actions')
    ax.set_zlabel('Q value')
    plt.show()

def plot2(Q, a=0):
    # plotting the Q function just to see
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = [], [], []
    for s in range(50000):
        p = np.random.rand()
        if p < 0.1 and Q[s,a] != 0:
            x += [(s-1)%500]
            y += [(s-1)/500]
            z += [Q[s, a]]
    ax.scatter(x, y, z)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Q value')
    plt.show()



# file_name = "data/medium.csv"
# gamma = 1
# Q = sarsa(file_name, 50000, 7, niter=10000, gamma=gamma)
# save(Q, "result/medium_q.pkl")
# P = optimal_policy_from_Q(Q)
# write_policy_to_file(P, "medium.policy")

Q = load("result/medium_q_corn.pkl")
plot2(Q, a=3)

# Q = load("result/medium_q_approx")
# plot2(Q)
Q = approximate_Q(Q, 50000, 7)
plot2(Q, a=3)

# save(Q, "result/medium_q_approx")
# P = optimal_policy_from_Q(Q)
# write_policy_to_file(P, "medium_approx.policy")