#Code reference:
#https://github.com/wishesrepo/reinforcement_learning/blob/master/policy_gambler_1.ipynb

import matplotlib.pyplot as plt
import numpy as np
import random
size=500
import time

def one_step_lookahead(s,V, p_h = 0.4):
    rewards = np.zeros(size+1)
    rewards[size] = 1
    A = np.zeros(size+1)
    stakes = range(1,min(s, size-s)+1)
    for a in stakes:
        A[a] = p_h * (rewards[s+a] + V[s+a]) + (1-p_h) * (rewards[s-a] + V[s-a])
    return A


def evaluate_policy(policy, theta=0.0001, max_backups=1000):
    old_values = np.zeros(size+1)
    deltaV=[]
    for i in range(max_backups):
        delta = 0
        new_values = np.zeros(size+1)
        for s in range(1, size):
            action_values = one_step_lookahead(s, old_values)
            new_values[s] = action_values[int(policy[s])]
        delta=np.max(np.abs(new_values - old_values))
        deltaV.append(delta)
        if np.max(np.abs(new_values - old_values)) < theta:
            break

        old_values = new_values
    return new_values, deltaV

def greedy_policy(value_function):
    policy = np.zeros(size+1)
    for s in range(1,size):
        action_values = one_step_lookahead(s,value_function)
        policy[s] = np.argmax(action_values)
    #print("Policy this iteration is \n")
    #print(policy)
    return policy

def policy_iteration():
    old_policy = np.zeros(size+1)
    for i in range(100):
        value_function, deltaV = evaluate_policy(old_policy)
        #print(value_function)
        new_policy = greedy_policy(value_function)
        if np.array_equal(old_policy,new_policy):
            break
        old_policy = new_policy
    return old_policy, deltaV

def runPI():
    t0=time.time()
    Policy, deltaV = policy_iteration()
    t1=time.time()
    t=t1-t0
    print("Total run time for gambler PI: ", t)
    plt.figure()
    plt.tight_layout(True)
    plt.title("Gambler Policy Iteration Convergence")
    plt.xlabel("iterations")
    plt.ylabel("delta V")
    plt.grid(True)
    plt.savefig("Gambler PI Convergence")
    plt.plot(deltaV)
    plt.show()

    plt.figure()
    plt.plot(Policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')
    plt.title('Gambler Policy Iteration Capital vs Final Policy')
    plt.grid(True)
    plt.tight_layout(True)
    plt.savefig("Gambler PI Final Policy")
    plt.show()
