#CodeDraft source for this gambler.py:
#https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Gamblers%20Problem%20Solution.ipynb
import numpy as np
import sys
import matplotlib.pyplot as plt

size=500 #size for gambler environments


def value_iteration(p_h, theta=0.0001, discount_factor=1):
    """
    Args:
        p_h: Probability of the coin coming up heads
    """
    # The reward is zero on all transitions except those on which the gambler reaches his goal,
    # when it is +1.
    rewards = np.zeros(size+1)
    rewards[size] = 1

    # We introduce two dummy states corresponding to termination with capital of 0 and 100
    V = np.zeros(size+1)

    def one_step_lookahead(s, V, rewards):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            s: The gambler’s capital. Integer.
            V: The vector that contains values at each state.
            rewards: The reward vector.

        Returns:
            A vector containing the expected value of each action.
            Its length equals to the number of actions.
        """
        A = np.zeros(size+1)
        stakes = range(1, min(s, size - s) + 1)  # Your minimum bet is 1, maximum bet is min(s, 100-s).
        for a in stakes:
            # rewards[s+a], rewards[s-a] are immediate rewards.
            # V[s+a], V[s-a] are values of the next states.
            # This is the core of the Bellman equation: The expected value of your action is
            # the sum of immediate rewards and the value of the next state.
            A[a] = p_h * (rewards[s + a] + V[s + a] * discount_factor) + (1 - p_h) * (
                        rewards[s - a] + V[s - a] * discount_factor)
        return A
    delta_V=[]
    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(1, size):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V, rewards)
            #print(s,A,V)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10.
            V[s] = best_action_value
        delta_V.append(delta)
        if delta < theta:

            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros(size)
    for s in range(1, size):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V, rewards)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s] = best_action

    return policy, V, delta_V

def rungambler():
    policy, v, deltaV = value_iteration(0.4)

    plt.figure()
    plt.plot(deltaV)
    plt.tight_layout(True)
    plt.title("Gambler Value Iteration Convergence")
    plt.xlabel("iterations")
    plt.ylabel("delta V")
    plt.grid(True)
    plt.savefig("Gambler VI Convergence")
    plt.show()

    print("Optimized Policy:")
    print(policy)
    print("")

    print("Optimized Value Function:")
    print(v)
    print("")


    x = range(size)
    # corresponding y axis values
    y = v[:size]

    # plotting the points
    plt.plot(x, y)

    # naming the x axis
    plt.xlabel('Capital')
    # naming the y axis
    plt.ylabel('Value Estimates')

    # giving a title to the graph
    plt.title('Value Iteration Final Policy (action stake) vs State (Capital)')
    plt.tight_layout(True)
    plt.grid(True)
    plt.savefig("Gambler VI Value Estimates")
    # function to show the plot
    plt.show()

    # Plotting Capital vs Final Policy
    x = range(size)
    y = policy

    plt.bar(x, y, align='center', alpha=0.5)

    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')
    plt.title('Gambler Value Iteration Capital vs Final Policy')
    plt.tight_layout(True)
    plt.savefig("Gambler VI Final Policy")
    plt.show()