import numpy as np
import csv
from TicTacTom import gameboard

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Sigmoid:
    #Normalise to 0<x<1
    def forward(self, x):
        self.output = 1 / (1+ np.exp(-x))

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


def generate_dataset(games):
    for each in range(games):
        print("Generating a new game")
        game = gameboard()
        game.nextTurn()
        print(game.gamestate)


        #the desired output (0-9) should be the next move

class Network():
    def __init__(self):
        self.layer1 = Layer_Dense(9,729)
        self.layer2 = Layer_Dense(729,6561)
        self.layer3 = Layer_Dense(6561,729)
        self.layer4 = Layer_Dense(729,9)

        self.ReLU = Activation_ReLU()
        self.sigmoid = Activation_Sigmoid()

        self.dataset = []

    def train(self, path, iterations):
        #self.dataset
        with open('dataset.csv') as csvfile:
            dataset_reader = csv.reader(csvfile)

            self.dataset = list(dataset_reader)

            # for sample in self.dataset:
            #     print(', '.join(sample))


            for iteration in range(iterations):
                #Open dataset csv file with each game sequence separated by a new line /n separator
                for sample_n in range(len(self.dataset)):

                    for state_n in range(len(self.dataset[sample_n])):

                        #output = self.think(sample_n)
                        error = self.get_label(sample_n, state_n)# - output
                        #exit()
                        #layeronecost = np.dot(training_inputs.T, error * self.sigmoidDerivative(output))
                        #layertwocost = np.dot(training_inputs.T, error * self.sigmoidDerivative(output))
                        #layerthreecost = np.dot(training_inputs.T, error * self.sigmoidDerivative(output))
                        #self.layeroneweights += layeronecost
                        #self.layertwoweights += layertwocost
                        #self.layeroneweights += layerthreecost
                    break

    def get_label(self, sample_n, state_n):
        desired_index = [] #The desired output from the network (aka. the next decision the ai should make)

        if len(self.dataset[sample_n]) - 1 > state_n:
            a = np.array(self.dataset[sample_n][state_n+1])
            print(a)
            b = np.array(self.dataset[sample_n][state_n])
            print(b)

            #print(np.subtract(a,b))
            #print(result)
            
if __name__ == '__main__':

        #X is the current gamestate and y is the next move to make
        #X = np.random.random((1,9))
        #print(X)

        network = Network()

        #Train on the dataset
        network.train("./dataset.csv", 1)

        network.layer1.forward(X)
        network.ReLU.forward(layer1.output)

        network.layer2.forward(ReLU.output)
        network.ReLU.forward(layer2.output)

        networklayer3.forward(ReLU.output)
        network.ReLU.forward(layer3.output)

        network.layer4.forward(ReLU.output)
        network.sigmoid.forward(layer4.output)

        print(network.sigmoid.output)


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
        # training_inputs = generate_dataset(1)
        #
        # # np.array([[0,0,1],
        # #                             [1,1,1],
        # #                             [1,0,1],
        # #                             [0,1,1]])
        #
        #
        # #Wins/Losses
        # #training_outputs = np.array([[0,1,1,0]]).T
        #
        # # Train the neural network
        # #neural_network.train(training_inputs, training_outputs, 10000)
        #
        # A = str(input("Input 1: "))
        # B = str(input("Input 2: "))
        # C = str(input("Input 3: "))
        #
        # print("New situation: input data = ", A, B, C)
        # print("Output data: ")
        #print(neural_network.think(np.array([A, B, C])))
