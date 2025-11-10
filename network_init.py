import numpy as np

def initialize_network(hidden_layers, hidden_neurons, learning_rate, activation_function, use_bias):

    input_features = 5
    output_classes = 3
    network = {'weights': [], 'biases': [], 'layers': hidden_neurons, 
               'activation': activation_function, 'use_bias': use_bias, 'learning_rate': learning_rate}

    layer_sizes = [input_features] + hidden_neurons + [output_classes]

                       
    for i in range(len(layer_sizes)-1):
        # Small random numbers for weights
        w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
        b = np.random.randn(layer_sizes[i+1]) * 0.1 if use_bias else np.zeros(layer_sizes[i+1])
        network['weights'].append(w)
        network['biases'].append(b)


    return network
