import numpy as np

def initialize_network(hidden_layers, hidden_neurons, learning_rate, activation_function, use_bias):

    input_size = 5  
    output_size = 3 
    
    layer_sizes = [input_size] + hidden_neurons + [output_size]
    weights = []
    biases = []
    
    print(f"Layer sizes: {layer_sizes}")
    
    # Initialize weights and biases for each layer
    for i in range(len(layer_sizes) - 1):
        current_size = layer_sizes[i]
        next_size = layer_sizes[i + 1]
        
        # Random weights from -0.5 to 0.5
        W = (np.random.rand(next_size, current_size) - 0.5)
        weights.append(W)
        
        # ALWAYS ZERO BIASES
        b = np.zeros(next_size)
        biases.append(b)
        
        print(f"Layer {i}: {current_size} -> {next_size}, Weight shape: {W.shape}")
    
    network = {
        'weights': weights,
        'biases': biases,
        'activation': activation_function,
        'learning_rate': learning_rate,
        'use_bias': use_bias,
        'layers': len(weights)
    }
    
    print("Network initialization complete!")
    return network