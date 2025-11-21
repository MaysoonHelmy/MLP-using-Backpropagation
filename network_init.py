import numpy as np

def initialize_network(hidden_layers, hidden_neurons, learning_rate, activation_function, use_bias):
    input_size = 5  
    output_size = 3 
    layer_sizes = [input_size] + hidden_neurons + [output_size]
    weights = []
    biases = []
    
    for i in range(len(layer_sizes) - 1):
        current_size = layer_sizes[i]
        next_size = layer_sizes[i + 1]        
        W = (np.random.rand(next_size, current_size) - 0.5)
        weights.append(W)
        b = np.zeros(next_size)
        biases.append(b)
            
    network = {
        'weights': weights,
        'biases': biases,
        'activation': activation_function,
        'learning_rate': learning_rate,
        'use_bias': use_bias,
        'layers': len(weights)
    }
    
    return network