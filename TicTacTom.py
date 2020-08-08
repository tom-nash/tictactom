import numpy as np
import os
from NeuralNet import PerceptronNeuralNetwork

class gameboard():
    def __init__(self, seed=2):
        #np.random.seed(seed)
        self.debug = True
        self.gamestate = np.full((3,3), 0) #Initialise the board to be all ' ' characters
        self.gameboard = np.full((3,3), ' ') #Initiaise the graphical board
        self.actionOnPlayer = np.random.choice([1,2]) #1 = Player, 2 = AI

    def getPlayerStr(self):
        if self.actionOnPlayer == 1:
            return "X"
        else:
            return "O"

    def switchPlayer(self):
        if self.actionOnPlayer == 1:
            self.actionOnPlayer = 2
        else:
            self.actionOnPlayer = 1

    def nextTurn(self):
        print("Enter the desired X, Y location on the board.")
        x = input("X:")
        y = input("Y:")

        if self.processTurn(int(x),int(y)):
            self.display()
        else:
            #turn invalid
            self.nextTurn()

        if not self.checkForWin():
            self.switchPlayer()
            self.nextTurn()
        else: #handle win
            print("Game over ", self.getPlayerStr(), " wins!")

    def processTurn(self, x, y):
        #Check that the selection is a valid location
        if self.checkSpot(x,y):
           self.gamestate[y-1][x-1] = self.actionOnPlayer
           self.gameboard[y-1][x-1] = self.getPlayerStr()
           print(x,y," is a valid selection")
           return True
        else:
            print("Invalid selection. Try again")
            return False

    def display(self):
        clearScrean()
        if self.debug:
            print(self.gamestate)
        print(self.gameboard)

    def checkSpot(self, x,y):
        if self.gamestate[y-1][x-1] == 1 or self.gamestate[y-1][x-1] == 2:
            return False
        else:
            return True

    #Function to consider the board looking specifically for win conditions.
    def checkForWin(self):
        for indexes in win_indexes(3):
            if all(self.gamestate[r][c] == self.actionOnPlayer for r, c in indexes):
                return True
        return False

def win_indexes(n):
    # Rows
    for r in range(n):
        yield [(r, c) for c in range(n)]
    # Columns
    for c in range(n):
        yield [(r, c) for r in range(n)]
    # Diagonal top left to bottom right
    yield [(i, i) for i in range(n)]
    # Diagonal top right to bottom left
    yield [(i, n - 1 - i) for i in range(n)]

def clearScrean():
    os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == '__main__':
    game = gameboard()
    NeuralNetwork = PerceptronNeuralNetwork(9,9)
    NeuralNetwork.gameboard = game

    clearScrean()
    print("Welcome to TicTacTom!\nThe first player is being chosen at random.")

    if game.actionOnPlayer is 1:
        print("The first player to take a turn is YOU")
    else:
        print("The AI gets the first turn")

    game.nextTurn()
