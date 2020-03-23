# multiAgents.py
# --------------
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

# NAME: VINCENT MAI
# DATE: 2/25/2020

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        "*** YOUR CODE HERE ***"
        
        foodList = currentGameState.getFood().asList() #consider food in current position of Pacman
        foodDistance = list()
        
        # Consider Manhattan Distance for Food
        # We want to Pacman to collect food and avoid the Ghost

        for currentGhost in newGhostStates:
            currentGPos = currentGhost.getPosition()
            currentTime = currentGhost.scaredTimer
            if (currentGPos == newPos and currentTime == 0):
                return -float("inf")
        
        for f in foodList:
            foodDistance.append(-1*manhattanDistance(f, newPos))
        
        return max(foodDistance)

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
       
        # index 0 in "agents" is Pacman
        # the rest are ghosts
        def terminal_state(state,  depth):
            # Terminal Conditions: if Win, Lost, or depth reached is top
            if state.isLose() or state.isWin() or self.depth == depth:
                return True
            else:
                return False
        
        def max_value(state, depth, ghost_count):
            # This method defines Pacman's actions
            if terminal_state(state, depth) == True:
                return self.evaluationFunction(state)
            
            pacman_actions = state.getLegalActions() # function to agent 0 --> pacman ; to obtain list of legal actions
            v = float("-inf") #set this variable to negative infinity
            start_ghost_index = 1
            for a in pacman_actions:
                v = max(v, min_value(state.generateSuccessor(0, a), depth, start_ghost_index, ghost_count)) 
            return v
        
        def min_value(state, depth, currentGhostIndex, ghost_count):
            # This methods defines the ghosts actions
            if terminal_state(state, depth) == True:
                return self.evaluationFunction(state)
            
            v = float("inf") # set v in min to infinity
            ghost_actions = state.getLegalActions(currentGhostIndex)
            if currentGhostIndex == ghost_count:
                for a in ghost_actions:
                    # evaluated all ghosts/mins so go back to the top to evaluate the max between these mins for Pacman's action
                    v = min(v, max_value(state.generateSuccessor(currentGhostIndex,a), depth+1, ghost_count))
            else:
                for a in ghost_actions:
                    # still need to evaluate this agent(s)
                    # stay in same level/depth but evaluate next ghost's actions
                    v = min(v, min_value(state.generateSuccessor(currentGhostIndex,a), depth, currentGhostIndex+1, ghost_count))

            # Return min value for max to choose max option against all other option
            return v
            
        
        # GOAL of Minimax return the action such that the max is taken from all the min values from corresponding agents
        # START evaluating at the bottom tree to find minimum of each then get the max out of all that
        scoreList = list()
        scoreList.append((float("-inf"), 'Stop')) # Worst Case, to stop
        ghost_count = gameState.getNumAgents() - 1
        score = float("-inf")
        for a in gameState.getLegalActions():
            score = min_value(gameState.generateSuccessor(0, a), 0, 1, ghost_count)
            score_action = (score,a)
            scoreList.append(score_action)


        sortedMax = sorted(scoreList, key=lambda tup: tup[0], reverse=True)
        ms, ma = sortedMax[0]
        return ma


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Implementation: Reusing the Code on MinimaxAgent and adding alpha and beta parameters
        # Such that for any max calculation, if the score is higher than beta, then automatically return
        # that action, alpha updates to be the max out of the score
        # For min calculation, the first action that is less than alpha will be returned
        # beta updates to be the min between the current score calculated and the current beta

        def terminal_state(state,  depth):
            # Terminal Conditions: if Win, Lost, or depth reached is top
            if state.isLose() or state.isWin() or self.depth == depth:
                return True
            else:
                return False
        
        def max_value(state, depth, ghost_count, alpha, beta):
            # This method defines Pacman's actions
            if terminal_state(state, depth) == True:
                return self.evaluationFunction(state)
            
            pacman_actions = state.getLegalActions() # function to agent 0 --> pacman ; to obtain list of legal actions
            v = float("-inf") #set this variable to negative infinity
            start_ghost_index = 1
            for a in pacman_actions:
                v = max(v, min_value(state.generateSuccessor(0, a), depth, start_ghost_index, ghost_count, alpha, beta)) 
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
        
        def min_value(state, depth, currentGhostIndex, ghost_count, alpha, beta):
            # This methods defines the ghosts actions
            if terminal_state(state, depth) == True:
                return self.evaluationFunction(state)
            
            v = float("inf") # set v in min to infinity
            ghost_actions = state.getLegalActions(currentGhostIndex)
            if currentGhostIndex == ghost_count:
                for a in ghost_actions:
                    # evaluated all ghosts/mins so go back to the top to evaluate the max between these mins for Pacman's action
                    v = min(v, max_value(state.generateSuccessor(currentGhostIndex,a), depth+1, ghost_count, alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
            else:
                for a in ghost_actions:
                    # still need to evaluate this agent(s)
                    # stay in same level/depth but evaluate next ghost's actions
                    v = min(v, min_value(state.generateSuccessor(currentGhostIndex,a), depth, currentGhostIndex+1, ghost_count, alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
            # Return min value for max to choose max option against all other option
            return v

        # GOAL of Minimax return the action such that the max is taken from all the min values from corresponding agents
        # START evaluating at the bottom tree to find minimum of each then get the max out of all that
        scoreList = list()
        scoreList.append((float("-inf"), 'Stop')) # Worst Case, to stop
        ghost_count = gameState.getNumAgents() - 1
        score = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        for a in gameState.getLegalActions():
            score = min_value(gameState.generateSuccessor(0, a), 0, 1, ghost_count, alpha, beta)
            score_action = (score,a)
            scoreList.append(score_action)
            if score > beta:
                return a
            alpha = max(alpha, score)

        sortedMax = sorted(scoreList, key=lambda tup: tup[0], reverse=True)
        ms, ma = sortedMax[0]
        return ma
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # index 0 in "agents" is Pacman
        # the rest are ghosts
        def terminal_state(state,  depth):
            # Terminal Conditions: if Win, Lost, or depth reached is top
            if state.isLose() or state.isWin() or self.depth == depth:
                return True
            else:
                return False
        
        def max_value(state, depth, ghost_count):
            # This method defines Pacman's actions
            if terminal_state(state, depth) == True:
                return self.evaluationFunction(state)
            
            pacman_actions = state.getLegalActions() # function to agent 0 --> pacman ; to obtain list of legal actions
            v = float("-inf") #set this variable to negative infinity
            start_ghost_index = 1
            for a in pacman_actions:
                v = max(v, expect_value(state.generateSuccessor(0, a), depth, start_ghost_index, ghost_count)) 
            return v
        
        def expect_value(state, depth, currentGhostIndex, ghost_count):
            # This methods defines the ghosts actions
            if terminal_state(state, depth) == True:
                return self.evaluationFunction(state)
            
            v = 0 # set v to be 0 to be summed for expected value calculations
            ghost_actions = state.getLegalActions(currentGhostIndex)
            if currentGhostIndex == ghost_count:
                for a in ghost_actions:
                    # evaluated all ghosts/mins so go back to the top to evaluate the max between these mins for Pacman's action
                    v += max_value(state.generateSuccessor(currentGhostIndex,a), depth+1, ghost_count)
            else:
                for a in ghost_actions:
                    # still need to evaluate this agent(s)
                    # stay in same level/depth but evaluate next ghost's actions
                    v += expect_value(state.generateSuccessor(currentGhostIndex,a), depth, currentGhostIndex+1, ghost_count)
                    
            # Return EXPECTED VALUE CALCULATIONS. In this case, add all the scores up and divide by the num of total actions
            # This is taking the average value
            return v/len(ghost_actions)
            
        
        # GOAL of Minimax return the action such that the max is taken from all the min values from corresponding agents
        # START evaluating at the bottom tree to find minimum of each then get the max out of all that
        scoreList = list()
        scoreList.append((float("-inf"), 'Stop')) # Worst Case, to stop
        ghost_count = gameState.getNumAgents() - 1
        score = float("-inf")
        for a in gameState.getLegalActions():
            score = expect_value(gameState.generateSuccessor(0, a), 0, 1, ghost_count)
            score_action = (score,a)
            scoreList.append(score_action)


        sortedMax = sorted(scoreList, key=lambda tup: tup[0], reverse=True)
        ms, ma = sortedMax[0]
        return ma

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    So basically what I did was reevaluate my reflex evaluation function as listed in the top of this code file.
    The implementation purely prioritize ending the game by collecting the food -- why? because if it stalls 
    then time runs on and score decreases. The only difference I've made in this evaluation function is that 
    this function ONLY considers the current game state and not the successors and the actions to get there.

    Thus, the getPacmanPosition() was the method I called on the currentGameState to analyze the environment relative to
    Pacman. With this, the manhattan distance implementation was also used for food. I was able to get an average of 1086.1 as
    my score for this question. What I also noticed was that when I added consideration for pacman distance to the ghosts, my
    scores fell significantly and that Pacman started to lose. Therefore, I took that out and thus only considered the most
    important aspect of the game for Pacman, to eat all the food so that the game could end with Pacman victorious.
    """

    "*** YOUR CODE HERE ***"
    
    
    # **************************************METHOD RUNS HERE****************************************************
    
    score = scoreEvaluationFunction(currentGameState) #Used for adversary search agents, not reflex

    foodList = currentGameState.getFood().asList() #consider food in current position of Pacman
    minFoodScore = float("inf")
    
    # Consider Manhattan Distance for Food
    
    for f in foodList:
        curDist = manhattanDistance(f, currentGameState.getPacmanPosition())
        if curDist < minFoodScore:
            minFoodScore = curDist
    
    score += (1/minFoodScore)
    return score


# Abbreviation
better = betterEvaluationFunction
