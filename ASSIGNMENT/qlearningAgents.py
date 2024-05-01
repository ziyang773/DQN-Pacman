# qlearningAgents.py
# ------------------
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


from game import *
from featureExtractors import *
from learningAgents import ReinforcementAgent
import util
import random

# my dependencies
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple, deque
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.q_values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if self.q_values[(state, action)]:
            return self.q_values[(state, action)]
        else:
            return 0.0
        # util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return 0.0
        tmp = util.Counter()
        for action in actions:
          tmp[action] = self.getQValue(state, action)
        return tmp[tmp.argMax()] 
        # util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return None
        
        best_action = None
        max_val = float("-inf")
        for action in actions:
          q_value = self.q_values[(state, action)]
          if max_val < q_value:
            max_val = q_value
            best_action = action
        return best_action
        # util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        explore = util.flipCoin(self.epsilon)
        if explore:
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # print(state.getPacmanState().getPosition())
        # if state.getPacmanState().getDirection()== "Stop":
        #     print(state.getPacmanState().getDirection())
        print(state.getPacmanState())
        print(state.getFood())
        print(state.getNumFood())
        print(state.getGhostPositions())
        print(getMatrix(state))
        print("+++++++++++++++")
        cur_q_value = self.getQValue(state, action)
        if nextState is None:
            self.q_values[(state, action)] = (1-self.alpha)*cur_q_value + self.alpha*reward
        else:
            self.q_values[(state, action)] = (1-self.alpha)*cur_q_value + self.alpha*(reward +self.discount*self.getValue(nextState))
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='SimpleExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        q_value = 0
        for i in features:
            q_value += features[i] * self.weights[i]
        return q_value
        # util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        # if len(features)>=4 :print(features)
        diff = reward + self.discount * self.getValue(nextState) - self.getQValue(state, action)
        for i in features:
            self.weights[i] += self.alpha * diff * features[i]
        # util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

###############################################################################################
####################################  My Work Below ###########################################
###############################################################################################
# MLP
class MLP(nn.Module):
    def __init__(self, lr, input_dimensions, output_dimensions):
        super(MLP, self).__init__()
        self.lr = lr
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions

        self.fc1 = nn.Linear(input_dimensions, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dimensions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# CNN  
class CNN(nn.Module):
    def __init__(self, lr, input_dimensions, output_dimensions):
        super(CNN, self).__init__()
        self.lr = lr
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(4, 16, 3, stride=1)
        self.fc1 = nn.Linear(16 * (input_dimensions[0]-4) * (input_dimensions[1]-4), 64)
        self.out = nn.Linear(64, output_dimensions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(-1)  # Flatten
        x = nn.functional.relu(self.fc1(x))
        x = self.out(x)
        return x
    
# Agent 1 with MLP  Need to be de-commentted 
class DQNAgent1(QLearningAgent):
    def __init__(self, extractor='SimpleExtractor', epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        # self.q_values = util.Counter()
        self.featExtractor = util.lookup(extractor, globals())()
        QLearningAgent.__init__(self, **args)
        self.input_size = 4
        self.action_size = 5
        self.model = MLP(self.alpha, self.input_size, self.action_size)
        self.action_list = ["Stop", "East", "West", "South", "North"]
        self.action_id = {"Stop":0, "East":1, "West":2, "South":3, "North":4}

    def update(self, state, action, nextState, reward):
        # this chunk this for MLP
        features = getFeatures(self.featExtractor.getFeatures(state, action))
        feature_vector = torch.tensor([features], dtype=torch.float32)
        current_q_values = self.model(feature_vector)[0, self.action_id[action]]
        reward_tensor = torch.tensor([reward], dtype=torch.float32)

        next_q_values_max = torch.tensor(-float("inf"), dtype=torch.float32)
        next_state_legalActions = self.getLegalActions(state)
        for legal_action in next_state_legalActions:
            next_state_features = getFeatures(self.featExtractor.getFeatures(nextState, legal_action))
            next_state_feature_vector=torch.tensor([next_state_features], dtype=torch.float32)
            next_q_values = self.model(next_state_feature_vector).max(1)[0].detach()
            next_q_values_max = max(next_q_values_max, next_q_values)

        self.model.optimizer.zero_grad()
        expected_q_values = reward_tensor + (self.discount * next_q_values_max)

        loss = self.model.loss(current_q_values.squeeze(), expected_q_values)
        loss.backward()
        self.model.optimizer.step()

    def getQValue(self, state, action):
        # this chunk is for MLP
        features = getFeatures(self.featExtractor.getFeatures(state, action))
        feature_vector = torch.tensor([features], dtype=torch.float32)
        with torch.no_grad():
            q_values = self.model(feature_vector)
            return q_values[0, action].item()

    def getAction(self, state):
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action
    
    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)
        if self.episodesSoFar == self.numTraining:
            pass

# Agent 1 with CNN need to be commeted out for MLP agent !!!
# !!!   
class DQNAgent1(QLearningAgent):
    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)
        self.input_dimensions = [8,7] # [8,7] for medium grid, [7,7] for small grid
        self.action_size = 5
        self.model = CNN(self.alpha, self.input_dimensions, self.action_size)
        self.action_list = ["Stop", "East", "West", "South", "North"]
        self.action_id = {"Stop":0, "East":1, "West":2, "South":3, "North":4}

    def update(self, state, action, nextState, reward):
        matrix = getMatrix(state)
        input_tensor = torch.tensor([matrix], dtype=torch.float32).unsqueeze(1)
        current_q_values = self.model(input_tensor)[self.action_id[action]]
        reward_tensor = torch.tensor([reward], dtype=torch.float32)

        next_state_matrix = getMatrix(nextState)
        next_state_input_tensor=torch.tensor([next_state_matrix], dtype=torch.float32).unsqueeze(1)
        next_q_values = self.model(next_state_input_tensor).max(0)[0].detach()

        self.model.optimizer.zero_grad()
        expected_q_values = reward_tensor + (self.discount * next_q_values)

        loss = self.model.loss(current_q_values.squeeze(), expected_q_values)
        loss.backward()
        self.model.optimizer.step()

    def getQValue(self, state, action):
        # this chunk is for MLP
        features = getFeatures(self.featExtractor.getFeatures(state, action))
        feature_vector = torch.tensor([features], dtype=torch.float32)
        with torch.no_grad():
            q_values = self.model(feature_vector)
            return q_values[0, action].item()

    def getAction(self, state):
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action
    
    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)
        if self.episodesSoFar == self.numTraining:
            pass

###############################################################################################
###############################################################################################
# Agent 2 with MLP Need to be de-commentted 
class DQNAgent2(QLearningAgent):
    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        # self.q_values = util.Counter()
        QLearningAgent.__init__(self, **args)
        self.input_size = 5
        self.action_size = 5
        self.model = MLP(self.alpha, self.input_size, self.action_size)
        self.action_list = ["Stop", "East", "West", "South", "North"]
        self.action_id = {"Stop":0, "East":1, "West":2, "South":3, "North":4}

    def update(self, state, action, nextState, reward):
        state_tensor = myFeatureExtractor(state)
        nextState_tensor = myFeatureExtractor(nextState)
        # action = torch.tensor([action], dtype=torch.int64)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)

        self.model.optimizer.zero_grad()
        current_q_values = self.model(state_tensor)[0, self.action_id[action]]
        next_q_values = self.model(nextState_tensor).max(1)[0].detach()
        expected_q_values = reward_tensor + (self.discount * next_q_values)

        loss = self.model.loss(current_q_values.squeeze(), expected_q_values)
        loss.backward()
        self.model.optimizer.step()

    def getQValue(self, state, action):
        state_tensor = myFeatureExtractor(state)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            return q_values[0, action].item()

    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        state_tensor = myFeatureExtractor(state)
        action = None
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            q_values = self.model(state_tensor).squeeze()
            _, indices = torch.sort(q_values, descending=True)
            indices = indices.tolist()
            for id in indices:
                action_temp = self.action_list[id]
                if action_temp in legalActions:
                    action = action_temp
                    break
        self.doAction(state, action)
        return action
    
    # def final(self, state):
    #     "Called at the end of each game."
    #     # call the super-class final method
    #     PacmanQAgent.final(self, state)

    #     # did we finish training?
    #     if self.episodesSoFar == self.numTraining:
    #         # you might want to print your weights here for debugging
    #         "*** YOUR CODE HERE ***"
    #         pass

# Agent 2 with CNN need to be commeted out for MLP agent !!!
# !!!
class DQNAgent2(QLearningAgent):
    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)
        self.input_dimensions = [8,7] # [8,7] for medium grid, [7,7] for small grid
        self.action_size = 5
        self.model = CNN(self.alpha, self.input_dimensions, self.action_size)
        self.action_list = ["Stop", "East", "West", "South", "North"]
        self.action_id = {"Stop":0, "East":1, "West":2, "South":3, "North":4}

    def update(self, state, action, nextState, reward):
        matrix = getMatrix(state)
        input_tensor = torch.tensor([matrix], dtype=torch.float32).unsqueeze(1)
        current_q_values = self.model(input_tensor)[self.action_id[action]]
        reward_tensor = torch.tensor([reward], dtype=torch.float32)

        next_state_matrix = getMatrix(nextState)
        next_state_input_tensor=torch.tensor([next_state_matrix], dtype=torch.float32).unsqueeze(1)
        next_q_values = self.model(next_state_input_tensor).max(0)[0].detach()

        self.model.optimizer.zero_grad()
        expected_q_values = reward_tensor + (self.discount * next_q_values)

        loss = self.model.loss(current_q_values.squeeze(), expected_q_values)
        loss.backward()
        self.model.optimizer.step()

    def getQValue(self, state, action):
        matrix = getMatrix(state)
        input_tensor = torch.tensor([matrix], dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            q_values = self.model(input_tensor)
            return q_values[action].item()

    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        matrix = getMatrix(state)
        input_tensor = torch.tensor([matrix], dtype=torch.float32).unsqueeze(1)
        action = None
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            q_values = self.model(input_tensor).squeeze()
            _, indices = torch.sort(q_values, descending=True)
            indices = indices.tolist()
            for id in indices:
                action_temp = self.action_list[id]
                if action_temp in legalActions:
                    action = action_temp
                    break
        self.doAction(state, action)
        return action
    
    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)


###############################################################################################
###############################################################################################
# helper functions   
def get_onehot(action):
    """
    Convert action to vector.
    """
    actions_onehot = np.zeros(5)
    if action == "Stop":
        actions_onehot[0] = 1
    elif action == "East":
        actions_onehot[1] = 1
    elif action == "West":
        actions_onehot[2] = 1
    elif action == "South":
        actions_onehot[3] = 1
    elif action == "North":
        actions_onehot[4] = 1
    return actions_onehot  

def getFeatures(features):
    """
    get features from SimpleExtractor.
    """
    features_vector = np.zeros(4)
    if len(features) == 3:
        features_vector[0] = features["bias"]
        features_vector[1] = features["#-of-ghosts-1-step-away"]
        features_vector[3] = features["closest-food"]
    else:
        features_vector[0] = features["bias"]
        features_vector[1] = features["#-of-ghosts-1-step-away"]
        features_vector[2] = features["eats-food"]
        features_vector[3] = features["closest-food"]

    return features_vector

def stateToTensor(state):
    """
    Convert state to tensor.
    """
    x, y = state.getPacmanState().getPosition()
    direction = state.getPacmanState().getDirection()
    direction = directionToInt(direction)
    state_tensor = torch.FloatTensor([x, y, direction])
    return state_tensor.unsqueeze(0)

def directionToInt(direction):
    """
    Convert direction to integer.
    """
    if direction == "Stop":
        return 0
    elif direction == "East":
        return 1
    elif direction == "West":
        return 2
    elif direction == "South":
        return 3
    elif direction == "North":
        return 4

def getMatrix(state):
    """
    Get matrix from state for CNN.
    """
    walls = state.getWalls()
    Pacman = state.getPacmanState().getPosition()
    ghost = state.getGhostPositions()[0]
    food = state.getFood()
    # get food position
    food_pos = []
    for i in range(walls.width):
        for j in range(walls.height):
            if food[i][j] == True:
                food_pos.append((i, j))

    matrix = []
    for j in range(walls.height):
        row = []
        for i in range(walls.width):
            if walls[i][walls.height-1-j] == True:
                row.append(-10)
            elif (i, walls.height-1-j) == Pacman:
                row.append(100)
            elif (i, walls.height-1-j) == ghost:
                row.append(-1000)
            elif (i, walls.height-1-j) in food_pos:
                row.append(1000)
            else:
                row.append(0)
        matrix.append(row)
    return matrix
    
def myFeatureExtractor(state):
    """
    My feature extractor from state, does not depend on action.
    """
    walls = state.getWalls()
    pacmanPos = state.getPacmanState().getPosition()
    score = state.getScore() 
    ghost = state.getGhostPositions()[0]
    food = state.getFood()
    # get food position
    food_dist = closestFood((pacmanPos[0], pacmanPos[1]), food, walls)
    if food_dist is not None:
        food_dist  = float(food_dist) 
    ghost_dist = util.manhattanDistance(pacmanPos, ghost)
    features = [pacmanPos[0], pacmanPos[1], food_dist, ghost_dist, score]
    features = torch.tensor([features], dtype=torch.float32)
    return features