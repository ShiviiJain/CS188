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


from util import manhattanDistance
from game import Directions
import random, util
from math import inf

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    
    def evaluationFunction(self, currentGameState: GameState, action):
            # Useful information you can extract from a GameState (pacman.py)
            successorGameState = currentGameState.generatePacmanSuccessor(action)
            newPos = successorGameState.getPacmanPosition()
            newFood = successorGameState.getFood()
            newGhostStates = successorGameState.getGhostStates()
            newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

            newX, newY = newPos
            scoree = currentGameState.getScore()
            i, j, k, score = 0, 0, 0, 0
            g = k
            fdist = []
            scaryTimes = newScaredTimes
            while i <= len(newGhostStates) - 1:
                ghostdist = manhattanDistance(newPos, newGhostStates[i].getPosition())
                i = i + 1

                if ghostdist == 1 or ghostdist <= 0:
                    score = score -  3
                    scoree = scoree - 3

                if newPos in currentGameState.getFood().asList():
                    score = score + 1
                    scoree = scoree + 1

                if currentGameState.hasWall(newX, newY) == True:
                    score = score - 4
                    scoree = scoree - 4

                while j <= len(currentGameState.getFood().asList()) - 1:
                    k = k + 2
                    g = g + 3
                    p, q = currentGameState.getFood().asList()[j]
                    fdist.append(abs(p - newX))
                    j = j + 1 
            return score * 1000/((2 * 5)*(1 * 10)) - min(fdist)/2

        

        

    


    

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        
        def minimax(gstate, depth, agent):
            if(depth == 0 or gstate.isWin() or gstate.isLose()):
                return self.evaluationFunction(gstate), None
            #if(self.depth == depth):
            #    return self.evaluationFunction(gstate)
            if(agent == 0):
                return maximize(gstate, depth, agent)
            else:
                #minimise
                return minimize(gstate, depth, agent)

        def minimize(gstate, depth, agent):
            finalaction = None
            actions = gstate.getLegalActions(agent)
            minval = float("inf")
            for i in actions:
            #    finalval = minimax(gstate.generateSuccessor(agent,i), depth2, agent2)
                if(agent == gstate.getNumAgents()-1):
                    finalval = minimax(gstate.generateSuccessor(agent,i), depth - 1, 0)[0]
                else:
                    finalval = minimax(gstate.generateSuccessor(agent,i),depth, agent + 1)[0]
                if (finalval < minval):
                    minval = finalval
                    finalaction = i
            return minval, finalaction

        def maximize(gstate, depth, agent):
            actions = gstate.getLegalActions(agent)
            maxval = -float("inf")
            for i in actions:
            #    finalval = minimax(gstate.generateSuccessor(agent,i), depth2, agent2)
                finalval = minimax(gstate.generateSuccessor(agent,i), depth, agent + 1)[0]
                if (finalval > maxval):
                    maxval = finalval
                    finalaction = i
            return maxval, finalaction
        final = minimax(gameState, self.depth, 0)
        actionend = final[1]
        return actionend
            
    


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphabeta(gstate, depth, agent, alpha, beta):
            if(depth == 0 or gstate.isWin() or gstate.isLose()):
                return self.evaluationFunction(gstate), None
            if(agent == 0):
                return maximizing_func(gstate, depth, agent, alpha, beta)
            else:
                return minimizing_func(gstate, depth, agent, alpha, beta)

        def minimizing_func(gstate, depth, agent, alpha, beta):
            finalaction = None
            actions = gstate.getLegalActions(agent)
            minval = float("inf")
            for i in actions:
                if(agent == gstate.getNumAgents()-1):
                    finalval = alphabeta(gstate.generateSuccessor(agent,i), depth - 1, 0, alpha, beta)[0]
                else:
                    finalval = alphabeta(gstate.generateSuccessor(agent,i),depth, agent + 1, alpha, beta)[0]
                if (finalval < minval):
                    minval = finalval
                    finalaction = i
                if (minval < alpha):
                    return minval, i
                beta = min(minval, beta)
            return minval, finalaction

        def maximizing_func(gstate, depth, agent, alpha, beta):
            actions = gstate.getLegalActions(agent)
            maxval = -float("inf")
            for i in actions:
                finalval = alphabeta(gstate.generateSuccessor(agent,i), depth, agent + 1, alpha, beta)[0]
                if (finalval > maxval):
                    maxval = finalval
                    finalaction = i
                if (maxval > beta):
                    return maxval, i
                alpha = max(maxval, alpha)
            return maxval, finalaction
        final = alphabeta(gameState, self.depth, 0, -float("inf"), float("inf"))
        actionend = final[1]
        return actionend




class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(gstate, depth, agent):
            if(depth == 0 or gstate.isWin() or gstate.isLose()):
                return self.evaluationFunction(gstate), None
            if(agent == 0):
                return maximizer(gstate, depth, agent)
            else:
                return exp(gstate, depth, agent)

        def exp(gstate, depth, agent):
            actions = gstate.getLegalActions(agent)
            finalval = 0
            for i in actions:
                if(agent == gstate.getNumAgents()-1):
                    finalval = finalval + expectimax(gstate.generateSuccessor(agent,i), depth - 1, 0)[0]
                else:
                    finalval = finalval + expectimax(gstate.generateSuccessor(agent,i),depth, agent + 1)[0]
            expval = finalval/len(actions)
            return expval, None
        
        def maximizer(gstate, depth, agent):
            actions = gstate.getLegalActions(agent)
            maxval = -float("inf")
            for i in actions:
                finalval = expectimax(gstate.generateSuccessor(agent,i), depth, agent + 1)[0]
                if (finalval > maxval):
                    maxval = finalval
                    finalaction = i
            return maxval, finalaction
        final = expectimax(gameState, self.depth, 0)
        actionend = final[1]
        return actionend


def betterEvaluationFunction(currentGameState: GameState):

    PacPos = currentGameState.getPacmanPosition()
    Foodlist = currentGameState.getFood().asList()
    CatFoodList = currentGameState.getFood().asList()
    ScaredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]
    score = currentGameState.getScore()
    scoree = currentGameState.getScore()
    nGDist = 0
    for ghost in currentGameState.getGhostStates():
        ghostPos = ghost.getPosition()
        distance = abs(PacPos[0] - ghostPos[0]) + abs(PacPos[1] - ghostPos[1]) 
        nGDist = min(nGDist, distance)

    foodDist = []
    for f in Foodlist:
        dist = abs(PacPos[0] - f[0]) + abs(PacPos[1] - f[1])
        foodDist.append(dist)
    totalfoodDist = sum(foodDist)
    
    if score == scoree and CatFoodList == Foodlist and totalfoodDist:
        if nGDist >= ScaredTimes[::-1][-1]:
            score = score + 0.7/min(foodDist)
        score = score + 1.1/ min(foodDist)
    return score

# Abbreviation
better = betterEvaluationFunction
