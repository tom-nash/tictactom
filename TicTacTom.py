import numpy as np
import csv
import random
import os
#from NeuralNet import PerceptronNeuralNetwork

class gameboard():
    def __init__(self, seed=2):
        #np.random.seed(seed)
        self.debug = True
        self.gamestate = np.full((3,3), 0) #Initialise the board to be all ' ' characters
        self.gameboard = np.full((3,3), ' ') #Initiaise the graphical board
        self.gameboardlist = []
        self.actionOnPlayer = np.random.choice([1,2]) #1 = Player, 2 = AI
        self.running = False

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
        # print("Enter the desired X, Y location on the board.")
        # x = input("X:")
        # y = input("Y:")
        self.processTurn()
        self.gameboardlist.append(self.gamestate.flatten().tolist())
        self.display()

    def processTurn(self):
        #Check that the selection is a valid location
        valid_selection = False
        while valid_selection == False:
            x = random.randrange(3)
            y = random.randrange(3)
            if self.checkSpot(x,y):
               self.gamestate[y-1][x-1] = self.actionOnPlayer
               self.gameboard[y-1][x-1] = self.getPlayerStr()
               valid_selection = True
        return True

    def checkSpot(self, x,y):
        #print(x,y)
        if self.gamestate[y-1][x-1] == 1 or self.gamestate[y-1][x-1] == 2:
            return False
        else:
            return True

    #Function to consider the board looking specifically for win conditions.
    def checkForWin(self):
        for indexes in win_indexes(3):
            if all(self.gamestate[r][c] == self.actionOnPlayer for r, c in indexes):
                game.running = False

                return 1
        if np.count_nonzero(self.gamestate == 0)== 0:
            game.running = False
            return 2
        return 0

    def display(self):
        clearScrean()
        if self.debug:
           print(self.gamestate.flatten())
        #print(self.gameboard)

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

def dumpData(boardlist):
    with open('dataset.csv', mode='a') as dataset:
        dataset_writer = csv.writer(dataset, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        dataset_writer.writerows([boardlist])


if __name__ == '__main__':

    #NeuralNetwork = PerceptronNeuralNetwork(9,9)
    #NeuralNetwork.gameboard = game

    #clearScrean()
    #print("Welcome to TicTacTom!\nThe first player is being chosen at random.")

    # if game.actionOnPlayer == 1:
    #     print("The first player to take a turn is YOU")
    # else:
    #     print("The AI gets the first turn")

    #game loop
    for i in range(100):
        game = gameboard()
        game.nextTurn()
        game.running = True
        while game.running:
            status = game.checkForWin()
            if status == 0:
                game.switchPlayer()
                game.nextTurn()
            elif status == 1:
                #export the list of gameboards and result to
                print("Game over ", game.getPlayerStr(), " wins!")
                if game.actionOnPlayer == 1: #Only save the games where player #1 wins!
                    dumpData(game.gameboardlist)
            elif status == 2:
                print("Game over. It's a tie")
                dumpData(game.gameboardlist)
