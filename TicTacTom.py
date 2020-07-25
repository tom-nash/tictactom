import numpy as np

#A single layer perceptron network consisting of only one set of weights
class PerceptronNeuralNetwork():
    def __init__(self, x, y):
        #self.learning_rate = 1-6
        
        np.random.seed(1)
        
        #w, h matrix
        #2 * the matrix size, then -1 to get the mean of 0 and -1 < x <1
        self.weights = 2 * np.random.random((x,y)) - 1
    
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

            cost = np.dot(training_inputs.T, error * self.sigmoidDerivative(output))

            self.weights += cost

    #Mult the input matrix with the weights to get the output
    def think(self, input):
        input = input.astype(float)

        output = self.sigmoid(np.dot(input, self.weights))
        return output

if __name__ == '__main__':

        neural_network = PerceptronNeuralNetwork(3,1)

        # The training set, with 4 examples consisting of 3
        # input values and 1 output value
        training_inputs = np.array([[0,0,1],
                                    [1,1,1],
                                    [1,0,1],
                                    [0,1,1]])

        training_outputs = np.array([[0,1,1,0]]).T

        # Train the neural network
        neural_network.train(training_inputs, training_outputs, 10000)


        A = str(input("Input 1: "))
        B = str(input("Input 2: "))
        C = str(input("Input 3: "))
        
        print("New situation: input data = ", A, B, C)
        print("Output data: ")
        print(neural_network.think(np.array([A, B, C]))) 
