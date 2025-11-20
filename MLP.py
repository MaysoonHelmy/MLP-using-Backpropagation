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
            q = np.tanh(x)
            return q

    def activate_derivative(self, x):
        if self.activation == "Sigmoid":
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        elif self.activation == "Hyperbolic Tangent":
            t = np.tanh(x)
            return 1 - t * t

    def forward(self, x):
        nets = []
        activations = [x]   # include input layer

        for i in range(len(self.weights)):

            net = np.dot(activations[-1], self.weights[i])
            if self.use_bias:
                net += self.biases[i]

            nets.append(net)

            if i == len(self.weights)-1:
                a = 1/(1+np.exp(-net))     # output layer activation = sigmoid
            else:
                if self.activation == "Sigmoid":
                    a = 1/(1+np.exp(-net))
                else:
                    a = np.tanh(net)

            activations.append(a)

        return activations, nets

    def backward(self, X, y, activations, nets):
        deltas = [0] * len(self.weights)
        
        # Output layer error (sigmoid derivative)
        error = y - activations[-1]
        deltas[-1] = error * (activations[-1] * (1 - activations[-1]))
        
        # Backpropagate errors
        for i in range(len(self.weights) - 2, -1, -1):
            error = np.dot(deltas[i + 1], self.weights[i + 1].T)
            if self.activation == "Sigmoid":
                deriv = activations[i+1] * (1 - activations[i+1])
            else:  # Hyperbolic Tangent
                deriv = 1 - activations[i+1] ** 2
            deltas[i] = error * deriv

        # Update weights and biases (FIXED SHAPES)
        for i in range(len(self.weights)):
            # Ensure activations are 2D (batch_size, features)
            a_prev = activations[i].reshape(-1, 1) if len(activations[i].shape) == 1 else activations[i]
            
            # Reshape delta if needed
            delta = deltas[i].reshape(1, -1) if len(deltas[i].shape) == 1 else deltas[i]
            
            # Weight update
            weight_update = np.dot(a_prev.T, delta)
            self.weights[i] += self.learning_rate * weight_update
            
            if self.use_bias:
                # Sum deltas along batch dimension and reshape to match bias shape
                bias_update = np.sum(delta, axis=0, keepdims=True)
                self.biases[i] += self.learning_rate * bias_update.reshape(self.biases[i].shape)

    def train(self, X, y, epochs=100,training_mode="Stochastic"):
        for epoch in range(epochs):
            if training_mode == "Stochastic":
              for i in range(X.shape[0]):
                xi = X[i:i + 1] 
                yi = y[i:i + 1] 
                activations, inputs = self.forward(xi)  #activations> after activation fn / inputs > before activation fn
                self.backward(xi, yi, activations, inputs)
            elif training_mode == "Batch-only":
                activations, inputs = self.forward(X) 
                self.backward(X, y, activations, inputs)
  
    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            xi = X[i:i + 1]  
            activations, _ = self.forward(xi)
            output = activations[-1]
            predicted_class = np.argmax(output)
            predictions.append(predicted_class)
        return np.array(predictions)
