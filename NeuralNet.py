import numpy as np

#A single layer perceptron network consisting of only one set of weights
class PerceptronNeuralNetwork():
    def __init__(self, x, y):
        self.gameboard = None

        #self.learning_rate = 1-6

        np.random.seed(1)

        #w, h matrix
        #2 * the matrix size, then -1 to get the mean of 0 and -1 < x <1

        self.layeroneweights = 2 * np.random.random((9,1)) - 1 #9 nodes
        self.layertwoweights = 2 * np.random.random((729,1)) - 1 #81 nodes
        self.layerthreeweights = 2 * np.random.random((6561,1)) - 1 #729 nodes
        self.layerfourweights = 2 * np.random.random((729,1)) - 1 #81 nodes
        self.layerfiveweights = 2 * np.random.random((9,1)) - 1 #9 nodes

        print(self.layeroneweights)
        print(self.layertwoweights)
        print(self.layerthreeweights)
        print(self.layerfourweights)
        print(self.layerfiveweights)

    #Normalise to 0<x<1
    def sigmoid(self, x):
        return 1 / (1+ np.exp(-x))

    #Calculate the derivative of sigmoid for cost calculation
    def sigmoidDerivative(self, x):
        return x * (1 - x)

    #Iterate over samples, calculate a cost for each and tweak weights to
    #converge on a minimum
    def train(self, training_inputs, training_labels, iterations):
        for iteration in range(iterations):

            output = self.think(training_inputs)

            error = training_labels - output

            layeronecost = np.dot(training_inputs.T, error * self.sigmoidDerivative(output))
            layertwocost = np.dot(training_inputs.T, error * self.sigmoidDerivative(output))
            layerthreecost = np.dot(training_inputs.T, error * self.sigmoidDerivative(output))
            self.layeroneweights += layeronecost
            self.layertwoweights += layertwocost
            self.layeroneweights += layerthreecost

    #Mult the input matrix with the weights to get the output
    def think(self, input):
        input = input.astype(float)

        #First layer activation function is sigmoid
        first_layer_output = self.sigmoid(np.dot(input, self.layeroneweights))

        second_layer_output = self.sigmoid(np.dot(first_layer_output, self.layertwoweights))

        third_layer_output = self.sigmoid(np.dot(second_layer_output, self.layerthreeweights))

        return output

if __name__ == '__main__':

        neural_network = PerceptronNeuralNetwork(3,1)


        #The game boards 9x1 * No. of samples
        # The training set, with 4 examples consisting of 3
        # input values and 1 output value

        #I need a way of generating 9x1 game board permutations which lead to a win or loss.
        #
        #   Generate random game moves which lead to wins, pass each win as a 0 label attached to the state of
        #   the board at each move.
        #
        #   - The game must be run with random valid moves made until a win condition is reached.
        #   - the game must append to a text file each move of a win game.
        #   - The AI always gets O (Naughts)
        #   - 
        #
        training_inputs = np.array([[0,0,1],
                                    [1,1,1],
                                    [1,0,1],
                                    [0,1,1]])


        #Wins/Losses
        training_outputs = np.array([[0,1,1,0]]).T

        # Train the neural network
        neural_network.train(training_inputs, training_outputs, 10000)

        A = str(input("Input 1: "))
        B = str(input("Input 2: "))
        C = str(input("Input 3: "))

        print("New situation: input data = ", A, B, C)
        print("Output data: ")
        print(neural_network.think(np.array([A, B, C])))
