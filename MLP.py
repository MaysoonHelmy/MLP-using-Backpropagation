import numpy as np

class MLP:
    def __init__(self, network):
        self.network = network
        self.weights = network['weights']
        self.biases = network['biases']
        self.activation = network['activation']
        self.learning_rate = network['learning_rate']
        self.use_bias = network['use_bias']
        self.layers = network['layers']

    def activate(self, x):
        if self.activation == "Sigmoid":
            q = 1/(1+np.exp(-x))
            return q
        elif self.activation == "Hyperbolic Tangent":
            q = (1-np.exp(-x))/(1+np.exp(-x))
            return q

    def activate_derivative(self, x):

        pass

    def forward(self, x):
        NET=[]
        OUTPUT=[]
        for i in range(len(self.weights)):
            net = np.dot(x, self.weights[i]) + self.biases[i]  # x shape: (1, input)
            NET.append(net)
            q = self.activate(net)
            OUTPUT.append(q)
            x=q
        return OUTPUT,NET


    def backward(self, X, y, activations, inputs):
        pass

    def train(self, X, y, epochs=100,training_mode="Stochastic"):
        for epoch in range(epochs):
            # Loop over each sample individually (SGD)
            if training_mode == "Stochastic":
              for i in range(X.shape[0]):
                xi = X[i:i + 1]  # Take a single row (sample)
                yi = y[i:i + 1]  # Take the corresponding target
                activations, inputs = self.forward(xi)  #activations> after activation fn / inputs > before activation fn
                self.backward(xi, yi, activations, inputs)
            elif training_mode == "Batch-only":
                activations, inputs = self.forward(X)  # activations> after activation fn / inputs > before activation fn
                self.backward(X, y, activations, inputs)
    def predict(self, X):
        pass
