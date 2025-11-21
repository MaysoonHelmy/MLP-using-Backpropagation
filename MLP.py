import numpy as np

class MLP:
    def __init__(self, network):
        self.weights = [w.copy() for w in network['weights']]
        self.biases = [b.copy() for b in network['biases']]
        self.activation_type = network['activation']
        self.learning_rate = network['learning_rate']
        self.use_bias = network['use_bias']

    def _activate(self, x):
        if self.activation_type == "Sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation_type == "Hyperbolic Tangent":
            return np.tanh(x)

    def _activate_derivative(self, x):
        if self.activation_type == "Sigmoid":
            s = self._activate(x)
            return s * (1 - s)
        elif self.activation_type == "Hyperbolic Tangent":
            return 1 - np.tanh(x) ** 2

    def forward(self, x):
        x = np.asarray(x).flatten()
        activations = [x]  
        nets = []         

        for W, b in zip(self.weights, self.biases):
            net = W @ activations[-1]
            if self.use_bias:
                net += b

            nets.append(net)
            activations.append(self._activate(net))

        return nets, activations

    def backward_update(self, nets, activations, target):
        num_layers = len(self.weights)
        deltas = [None] * num_layers

        error = target - activations[-1]
        deltas[-1] = error * self._activate_derivative(nets[-1])

        for layer in range(num_layers - 2, -1, -1):
            next_W = self.weights[layer + 1].T
            deltas[layer] = self._activate_derivative(nets[layer]) * (next_W @ deltas[layer + 1])

        # Update weights
        for i in range(num_layers):
            grad_W = np.outer(deltas[i], activations[i])
            self.weights[i] += self.learning_rate * grad_W

            if self.use_bias:
                self.biases[i] += self.learning_rate * deltas[i]

    def train(self, X, y, epochs=100):
        X = X.values if hasattr(X, 'values') else X
        y = y.values if hasattr(y, 'values') else y
        y = y.astype(int)
        y_onehot = np.eye(3)[y]

        for _ in range(epochs):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y_onehot[indices]

            for xi, yi in zip(X_shuffled, y_shuffled):
                nets, activations = self.forward(xi)
                self.backward_update(nets, activations, yi)

    def predict(self, X):
        X = X.values if hasattr(X, 'values') else X
        preds = []
        
        for xi in X:
            _, activations = self.forward(xi)
            preds.append(np.argmax(activations[-1]))

        return np.array(preds)
