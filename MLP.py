import numpy as np

class MLP:
    def __init__(self, network):
        self.weights = [w.copy() for w in network['weights']]
        self.biases = [b.copy() for b in network['biases']]
        self.activation = network['activation']
        self.learning_rate = network['learning_rate']
        self.use_bias = network['use_bias']

    def _activate(self, x):
        if self.activation == "Sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation == "Hyperbolic Tangent":
            return np.tanh(x)

    def _activate_prime(self, x):
        if self.activation == "Sigmoid":
            s = self._activate(x)
            return s * (1 - s)
        elif self.activation == "Hyperbolic Tangent":
            return 1 - np.tanh(x) ** 2

    def forward(self, x):
        x = np.asarray(x).flatten()
        activations = [x]
        nets = []
        
        for i in range(len(self.weights)):
            net = np.dot(self.weights[i], activations[-1])
            if self.use_bias:
                net += self.biases[i]
            nets.append(net)
            activation = self._activate(net)
            activations.append(activation)
            
        return nets, activations

    def backward_update(self, nets, activations, target):
        deltas = [None] * len(self.weights)
        n_layers = len(self.weights)
        
        # Output layer
        output_error = target - activations[-1]
        deltas[-1] = output_error * self._activate_prime(nets[-1])
        
        # Hidden layers
        for i in range(n_layers - 2, -1, -1):
            sum_term = np.dot(self.weights[i + 1].T, deltas[i + 1])
            deltas[i] = self._activate_prime(nets[i]) * sum_term

        # Weight update
        for i in range(n_layers):
            grad_W = np.outer(deltas[i], activations[i])
            self.weights[i] += self.learning_rate * grad_W
            
            if self.use_bias:
                self.biases[i] += self.learning_rate * deltas[i]

    def train(self, X, y, epochs=100):
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        y = y.astype(int)
        y_onehot = np.eye(3)[y]
        
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y_onehot[indices]
            
            for i in range(X.shape[0]):
                xi = X_shuffled[i]
                yi = y_shuffled[i]
                
                nets, activations = self.forward(xi)
                self.backward_update(nets, activations, yi)

    def predict(self, X):
        if hasattr(X, 'values'):
            X = X.values
            
        predictions = []
        for i in range(X.shape[0]):
            xi = X[i]
            _, activations = self.forward(xi)
            predicted_class = np.argmax(activations[-1])
            predictions.append(predicted_class)
        
        return np.array(predictions)