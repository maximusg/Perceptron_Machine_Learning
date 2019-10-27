import numpy as np

class Perceptron():
    def __init__(self, weights=7):
        self.syn_weights =0
        if type(weights)==int:
            self.syn_weights = np.random.rand(weights,1)
        else:
            self.syn_weights = weights
        print("percep init", self.syn_weights)    


    def sigmoid(self, x):
        x = np.clip( x, -500, 500 ) #prevents overflow
        return (1 / (1 + np.exp(-x)))

    def sigmoid_deriv(self, x):
        x = np.clip( x, -250, 250 ) #prevents overflow
        return np.exp(-x)/((1 + np.exp(-x))**2)

    def train(self, inputs=0, real_outputs=0, its=0, lr=0):
        
        if inputs==0: #if the input is zero, just return weights
            return self.syn_weights
        
        delta_weights = np.zeros((len(self.syn_weights),len(inputs)))
       
        print("train")
        for iteration in (range(its)):
            if iteration%50==0:
                print("iter: ", iteration, "weights: ", repr(self.syn_weights))

            # forward pass
            z = np.dot(inputs, self.syn_weights)
            activation = self.sigmoid(z)

            # back pass
            for i in range(len(inputs)):
                cost = (activation[i] - real_outputs[i])**2
                cost_prime = 2*(activation[i] - real_outputs[i])
                for n in range(len(self.syn_weights)):
                    delta_weights[n][i] = cost_prime * inputs[i][n] * self.sigmoid_deriv(z[i])
                    
            delta_avg = np.array([np.average(delta_weights, axis=1)]).T
            self.syn_weights = self.syn_weights - delta_avg*lr
        return self.syn_weights

    def results(self, inputs):
        return self.sigmoid(np.dot(inputs, self.syn_weights))

    def printWeights(self):
        print("weights",self.syn_weights)


if __name__ == "__main__":

    ts_input = np.array([[0,0,1,0],
                         [1,1,1,0],
                         [1,0,1,1],
                         [0,1,1,1],
                         [0,1,0,1],
                         [1,1,1,1],
                         [0,0,0,0]])

    ts_output = np.array([[0,1,1,0,0,1,0]]).T # First Value of Input = output

    testing_data = np.array([[0,1,1,0],
                             [0,0,0,1],
                             [0,1,0,0],
                             [1,0,0,1],
                             [1,0,0,0],
                             [1,1,0,0],
                             [1,0,1,0]])

    lr = 10 # Learning Rate
    steps = 10000
    perceptron = Perceptron() # Initialize a perceptron
    perceptron.train(ts_input, ts_output, steps, lr) # Train the perceptron

    results = []
    for x in (range(len(testing_data))):
        run = testing_data[x]
        trial = perceptron.results(run)
        results.append(trial.tolist())
    print("results")
    print(results)
    print(np.ravel(np.rint(results))) # View rounded results
    print(perceptron.syn_weights)