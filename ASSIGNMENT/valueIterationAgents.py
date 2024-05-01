# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import util
from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            states = self.mdp.getStates()

            # a temo dict to store value
            temp_counter = util.Counter()
            for state in states:
                max_val = float("-inf")
                for action in self.mdp.getPossibleActions(state):
                    q_value = self.computeQValueFromValues(state, action)
                    if q_value > max_val:
                        max_val = q_value
                    temp_counter[state] = max_val

            # update self.values
            self.values = temp_counter

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        action_prob_pairs = self.mdp.getTransitionStatesAndProbs(state, action)
        q_value = 0
        # search through all possible paths
        for next_state, prob in action_prob_pairs:
            reward = self.mdp.getReward(state, action, next_state)
            q_value += prob * (reward + self.discount * self.values[next_state])
        return q_value
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # initialize action and value
        best_action = None
        max_val = float("-inf")
        
        for action in self.mdp.getPossibleActions(state):
          q_value = self.computeQValueFromValues(state, action)
          if q_value > max_val:
            max_val = q_value
            best_action = action
        return best_action
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        num_states = len(states)
        
        for i in range(self.iterations):
          # which state to update
          id = i % num_states
          state = states[id]
          if not self.mdp.isTerminal(state):
            max_val = float("-inf")
            for action in self.mdp.getPossibleActions(state):
              q_value = self.computeQValueFromValues(state, action)
              if q_value >= max_val:
                    max_val = q_value
            self.values[state] = max_val


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Compute predecessors of all states.
        predecessors = {}
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for nextState, _ in self.mdp.getTransitionStatesAndProbs(state, action):
                        if nextState in predecessors:
                            predecessors[nextState].append(state)
                        else:
                            predecessors[nextState] = [state]

        # Initialize an empty priority queue.
        priority_queue = util.PriorityQueue()

        # For each non-terminal state
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                # Find the absolute value of the difference between the current value of s in self.values 
                # and the highest Q-value across all possible actions from s
                cur_value = self.values[state]
                max_q = float("-inf")
                for action in self.mdp.getPossibleActions(state):
                    q_value = self.computeQValueFromValues(state, action)
                    max_q = max(max_q, q_value)
                diff = abs(max_q - cur_value)
                # Push s into the priority queue with priority -diff
                priority_queue.update(state, - diff)


        # For iteration in 0, 1, 2, ..., self.iterations - 1
        for i in range(self.iterations):
            # If the priority queue is empty, then terminate.
            if priority_queue.isEmpty():
                break

            # Pop a state s off the priority queue.
            state = priority_queue.pop()
            # Update the value of s (if it is not a terminal state) in self.values.
            if not self.mdp.isTerminal(state):
                max_val = float("-inf")
                for action in self.mdp.getPossibleActions(state):
                    q_value = self.computeQValueFromValues(state, action)
                    max_val = max(max_val, q_value)
                self.values[state] = max_val

            # For each predecessor p of s
            for predecessor in predecessors[state]:
                if not self.mdp.isTerminal(predecessor):
                    # Find the absolute value of the difference between the current value of s in self.values 
                    # and the highest Q-value across all possible actions from s
                    max_q = float("-inf")
                    for action in self.mdp.getPossibleActions(predecessor):
                        q_value = self.computeQValueFromValues(predecessor, action)
                        max_q = max(max_q, q_value)
                    diff = abs(max_q - self.values[predecessor])

                    # If diff > theta, push p into the priority queue with priority -diff 
                    if diff > self.theta:
                        priority_queue.update(predecessor, -diff)
