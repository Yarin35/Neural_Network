import math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return max(0.0, x)

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0

def softmax(vector):
    max_val = max(vector)
    exps = [math.exp(x - max_val) for x in vector]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def activate(vector, activation):
    if activation == "relu":
        return [relu(x) for x in vector]
    elif activation == "sigmoid":
        return [sigmoid(x) for x in vector]
    elif activation == "softmax":
        return softmax(vector)
    else:
        return vector

def activate_derivative(vector, activation):
    if activation == "relu":
        return [relu_derivative(x) for x in vector]
    elif activation == "sigmoid":
        return [sigmoid_derivative(x) for x in vector]
    else:
        return [1.0] * len(vector)

def matrix_vector_mult(matrix, vector):
    """Matrix (output_size x input_size) * vector (input_size) -> result (output_size)"""
    result = []
    for row in matrix:
        val = sum(w * v for w, v in zip(row, vector))
        result.append(val)
    return result

def vector_add(v1, v2):
    return [a + b for a, b in zip(v1, v2)]

def forward_pass(network, input_vector):
    """Returns (output, cache) where cache contains all activations"""
    cache = {"activations": [input_vector], "z_values": []}
    
    current = input_vector
    
    for i, layer in enumerate(network["layers"]):
        weights = network["weights"][i]
        biases = network["biases"][i]
        activation = layer["activation"]
        
        z = vector_add(matrix_vector_mult(weights, current), biases)
        cache["z_values"].append(z)
        
        current = activate(z, activation)
        cache["activations"].append(current)
    
    return current, cache

def cross_entropy_loss(predicted, target):
    """Cross-entropy loss for classification"""
    epsilon = 1e-15
    loss = -sum(t * math.log(max(p, epsilon)) for t, p in zip(target, predicted))
    return loss

def backward_pass(network, cache, target, learning_rate):
    """Backpropagation with gradient descent"""
    layers = network["layers"]
    weights = network["weights"]
    biases = network["biases"]
    
    activations = cache["activations"]
    z_values = cache["z_values"]
    
    num_layers = len(layers)
    
    # Output layer gradient
    output = activations[-1]
    delta = [o - t for o, t in zip(output, target)]
    
    # Backpropagate
    for i in range(num_layers - 1, -1, -1):
        prev_activation = activations[i]
        
        # Update weights and biases
        for j in range(len(weights[i])):
            for k in range(len(weights[i][j])):
                weights[i][j][k] -= learning_rate * delta[j] * prev_activation[k]
        
        for j in range(len(biases[i])):
            biases[i][j] -= learning_rate * delta[j]
        
        # Propagate error to previous layer
        if i > 0:
            next_delta = [0.0] * len(prev_activation)
            for j in range(len(prev_activation)):
                error = sum(weights[i][k][j] * delta[k] for k in range(len(delta)))
                activation_func = layers[i-1]["activation"] if i > 0 else None
                if activation_func:
                    deriv = activate_derivative(z_values[i-1], activation_func)
                    next_delta[j] = error * deriv[j]
                else:
                    next_delta[j] = error
            delta = next_delta
