import numpy as np
import os
class gameboard():
    def __init__(self, seed=2):
        #np.random.seed(seed)
        self.debug = False
        self.gamestate = np.full((3,3), 0) #Initialise the board to be all ' ' characters
        self.gameboard = np.full((3,3), ' ') #Initiaise the graphical board
        self.actionOnPlayer = np.random.choice([1,2]) #1 = Player, 2 = AI
        
    def getPlayerStr(self):
        if self.actionOnPlayer == 1:
            return "X"
        else:
            return "O"

    def nextTurn(self):
        print("Enter the desired X, Y location on the board.")
        x = input("X:")
        y = input("Y:")

        self.processTurn(int(x),int(y)) 
        self.display()

    def processTurn(self, x, y):
        #Check that the selection is a valid location
        if True:#self.checkSpot(player, i)
           self.gamestate[y-1][x-1] = self.actionOnPlayer  
           self.gameboard[y-1][x-1] = self.getPlayerStr()  

    def display(self):
        if self.debug:
            print(self.gamestate)
        print(self.gameboard) 

def clearScrean():
    os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == '__main__':
    game = gameboard()
    clearScrean()
    print("Welcome to TicTacTom!\nThe first player is being chosen at random.")
    if game.actionOnPlayer is 1:
        print("The first player to take a turn is YOU")
    else:
        print("The AI gets the first turn")

    game.nextTurn()


