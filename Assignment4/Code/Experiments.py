import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import FrozenLake_PIVI as FL
import gym
import time
import gambler_VI
import FrozenLake_Q
import gambler_PI

env=gym.make('FrozenLake-v0')
print (env)

#Frozen Lake Policy Iteration

t0=time.time()
PI_policy, PI_V, PI_deltaV = FL.policy_iteration(env)
t1=time.time()
PI_time=t1-t0
print (PI_V.reshape(4,4))
V1=PI_V.reshape(4,4)

sns.heatmap(V1, cmap="YlGnBu",annot=True, fmt='.2f')
plt.title("Frozen Lake Policy Iteration Optimal V")
plt.tight_layout(True)
plt.savefig("Frozen Lake PI optimal V.png",dpi=300)
plt.show()

#optimal policy
sns.heatmap(PI_policy.reshape(4,4), cmap="YlGnBu",annot=True)
plt.title("Frozen Lake Policy Iteration Optimal Policy")
plt.tight_layout(True)
plt.savefig("Frozen Lake PI Optimal Policy.png",dpi=300)
plt.show()

#Frozen Lake Value Iteration
t0=time.time()
VI_policy, VI_V, VI_deltaV=FL.value_iteration(env)
t1=time.time()
VI_time=t1-t0
print (VI_V.reshape(4,4))

sns.heatmap(VI_V.reshape(4,4), cmap="YlGnBu",annot=True,  fmt='.2f')
plt.title("Frozen Lake Value Iteration Optimal V")
plt.tight_layout(True)
plt.savefig("Frozen Lake VI optimal V.png",dpi=300)
plt.show()

#optimal policy
sns.heatmap(VI_policy.reshape(4,4), cmap="YlGnBu",annot=True)
plt.title("Frozen Lake Value Iteration Optimal Policy")
plt.tight_layout(True)
plt.savefig("Frozen Lake VI Optimal Policy.png",dpi=300)
plt.show()

print("Running time total in seconds for value iteration: ", VI_time)
print("Running time total in seconds for policy iteration: ", PI_time)

#Plot PI vs VI convergence
plt.figure()
plt.plot(VI_deltaV, color="blue", label="Value Iteration")
plt.plot(PI_deltaV, color="red", label="Policy Iteration")
plt.legend(loc="best")
plt.xlim()
plt.grid(True)
plt.xlabel("Iterations")
plt.ylabel("Delta V")
plt.title("Frozen Lake Convergence")
plt.tight_layout(True)
plt.savefig("Frozen Lake Convergence")
plt.show()

print (PI_policy)
print (VI_policy)

#Frozen Lake Q-learning
FrozenLake_Q.runQlearning()


#Gambler's problem
gambler_VI.rungambler()
gambler_PI.runPI()



