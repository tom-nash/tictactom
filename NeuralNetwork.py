import numpy as np
import csv
from TicTacTom import gameboard
from NeuralNet import NeuralNetwork

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        print(self.weights)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
    def reluDerivative(self, x):
        if x.any() > 0:
            return 1
        else:
            return 0

class Activation_Sigmoid:
    #Normalise to 0<x<1
    def forward(self, x):
        self.output = 1 / (1+ np.exp(-x))

    #Calculate the derivative of sigmoid for cost calculation
    def sigmoidDerivative(self, x):
        return x * (1 - x)

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
        #self.sigmoid = Activation_Sigmoid()

        self.learningrate = 1
        self.dataset = []

    def train(self, path, iterations):
        #self.dataset
        with open('dataset.csv') as csvfile:
            dataset_reader = csv.reader(csvfile)

            self.dataset = list(dataset_reader)

            for iteration in range(iterations):
                #Open dataset csv file with each game sequence separated by a new line /n separator
                for sample_n in range(len(self.dataset)):
                    for state_n in range(len(self.dataset[sample_n]) - 1):
                        next_label = self.get_next_label(sample_n, state_n)# - output

                        if np.count_nonzero(next_label == 1) > 0:
                            label = next_label.reshape(9,1)
                            state = np.array(self.dataset[sample_n][state_n][1:-1].split(','), dtype=np.int8)

                            if np.count_nonzero(state == 1) > 0:
                                #Perform inference on the state as it has a suitable label
                                self.layer1.forward(state)
                                self.ReLU.forward(self.layer1.output)
                                layeroneactivation = self.ReLU.output
                                self.layer2.forward(self.ReLU.output)
                                self.ReLU.forward(self.layer2.output)
                                layertwoactivation = self.ReLU.output
                                self.layer3.forward(self.ReLU.output)
                                self.ReLU.forward(self.layer3.output)
                                layerthreeactivation = self.ReLU.output
                                self.layer4.forward(self.ReLU.output)
                                self.ReLU.forward(self.layer4.output)
                                layerfouractivation = self.ReLU.output

                                output = self.ReLU.output
                                print("Found the label for AI", next_label, " Inferencing on State: ", state, " Output of network on state: ",output)

                                error = np.subtract(label, output)
                                #print("Calculated error on output: ", error)

                                #layeronecost = np.dot(label.T, error * self.ReLU.reluDerivative(self.layer1.output))
                                #layertwocost = np.dot(label.T, error * self.ReLU.reluDerivative(self.layer2.output))
                                #layerthreecost = np.dot(label.T, error * self.ReLU.reluDerivative(self.layer3.output))

                                # print(error.T * self.ReLU.reluDerivative(self.layer4.output))

                                #ReLU Derivative is not providing a NP Array! have it return an actual array otherwise error gets matmul by 1 or 0 instead of an array of 1s or 0s
                                #Back propagate the label to get the cost per layer

                                layerfourcost = np.dot(label.T, error * self.ReLU.reluDerivative(layerfouractivation))

                                layerthreecost = np.dot(label.T, error * self.ReLU.reluDerivative(layerthreeactivation))

                                layertwocost = np.dot(label.T, error * self.ReLU.reluDerivative(layertwoactivation))
                                print(layertwocost)

                                layeronecost = np.dot(label.T, error * self.ReLU.reluDerivative(layeroneactivation))
                                print(layeronecost)




                                self.layer1.weights = self.layer1.weights + self.learningrate * layeronecost
                                self.layer2.weights = self.layer2.weights + self.learningrate * layertwocost
                                self.layer3.weights = self.layer3.weights + self.learningrate * layerthreecost
                                self.layer4.weights = self.layer4.weights + self.learningrate * layerfourcost
                                # for i in range(len(self.layer1.biases)):
                                #     self.layer1.biases[i] -= error.reshape(1,0) * self.learningrate
                                # self.layer2.biases -= error.T * self.learningrate
                                # self.layer3.biases -= error.T * self.learningrate
                                # self.layer4.biases -= error.T * self.learningrate
                        else:
                            #print("Found a label for an opponent")
                            continue
                    #break #Choosing to only do one loop for debug reasons


    def think(self, inputs):
        self.layer1.forward(inputs)
        self.ReLU.forward(self.layer1.output)

        self.layer2.forward(self.ReLU.output)
        self.ReLU.forward(self.layer2.output)

        self.layer3.forward(self.ReLU.output)
        self.ReLU.forward(self.layer3.output)

        self.layer4.forward(self.ReLU.output)
        self.ReLU.forward(self.layer4.output)

        #print(self.sigmoid.output)
        return self.ReLU.output

    def get_next_label(self, sample_n, state_n):
        desired_index = [] #The desired output from the network (aka. the next decision the ai should make)
        result=np.array([])
        if len(self.dataset[sample_n]) - 1 > state_n:
            A = np.array(self.dataset[sample_n][state_n+1][1:-1].split(','), dtype='int64')

            B = np.array(self.dataset[sample_n][state_n][1:-1].split(','), dtype='int64')

            #result = [a - b for a, b in zip(A, B)]
            result=np.subtract(A,B)
        return(result) #This is the next choice

            # output=[]
            # for i in result:
            #     if i == 1: #Check that is it an AI Turn again? AI=#1
            #         print("AI TURN FOUND")

class test():
    def __init__(self):
        self.dataset = []
        self.X = []
        self.y = []

    def getDataset(self, iterations):
        #self.dataset
        with open('dataset.csv') as csvfile:
            dataset_reader = csv.reader(csvfile)

            self.dataset = list(dataset_reader)

            for iteration in range(iterations):
                #Open dataset csv file with each game sequence separated by a new line /n separator
                for sample_n in range(len(self.dataset)):
                    for state_n in range(len(self.dataset[sample_n]) - 1):
                        next_label = self.get_next_label(sample_n, state_n)# - output

                        if np.count_nonzero(next_label == 1) > 0:
                            label = next_label.reshape(9,1)
                            state = np.array(self.dataset[sample_n][state_n][1:-1].split(','), dtype=np.int8).reshape(9,1)

                            if np.count_nonzero(state == 1) > 0:
                                self.X.append(state)
                                self.y.append(label)


    def get_next_label(self, sample_n, state_n):
        desired_index = [] #The desired output from the network (aka. the next decision the ai should make)
        result=np.array([])
        if len(self.dataset[sample_n]) - 1 > state_n:
            A = np.array(self.dataset[sample_n][state_n+1][1:-1].split(','), dtype='int64')

            B = np.array(self.dataset[sample_n][state_n][1:-1].split(','), dtype='int64')

            #result = [a - b for a, b in zip(A, B)]
            result=np.subtract(A,B)
        return(result) #This is the next choice

if __name__ == '__main__':

        test = test()
        print("getting dataset")
        test.getDataset(1)
        print(np.asarray(test.y).reshape((9, -1)))
        print(np.shape(test.X))
        print(np.shape(test.y))

        nn = NeuralNetwork([9,18,18,9])
        nn.train(X=np.asarray(test.X).reshape((9, -1)), y=np.asarray(test.y).reshape((9, -1)), batch_size=9, epochs=2, learning_rate=0.4, print_every=10, validation_split=0.2, tqdm_=False, plot_every=20000)


        #X is the current gamestate and y is the next move to make
        #X = np.random.random((1,9))
        #print(X)

        #network = Network()

        #Train on the dataset
        #network.train("./dataset.csv", 3)



        #
        # #
        # X = np.array([0, 0, 0, 0, 1, 0, 1, 2, 2])
        #
        # print("Running inference on [0, 0, 0, 0, 1, 0, 1, 2, 2]. Expecting: [0,1,0,0,0,0,0,0,0]. Result: ",network.think(X))
        #
        # X = np.array([0, 2, 0, 1, 1, 2, 1, 2, 2])
        #
        # print("Running inference on [0, 0, 0, 0, 1, 0, 1, 2, 2]. Expecting: [0,1,0,0,0,0,0,0,0]. Result: ",network.think(X))

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
