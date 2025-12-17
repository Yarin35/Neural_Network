import json
import math
import random

def init_weights(input_size, output_size):
    limit = math.sqrt(6 / (input_size + output_size))
    return [[random.uniform(-limit, limit) for _ in range(input_size)] for _ in range(output_size)]

def init_biases(output_size):
    return [0.0] * output_size

def generate_network(name, config, n):
    base = name.replace(".conf", "")

    input_size = config["input_size"]
    layer_sizes = config["layer_sizes"]
    activations = config["activations"]
    learning_rate = float(config.get("learning_rate", 0.01))

    layers = []
    prev_size = input_size
    
    for size, activation in zip(layer_sizes, activations):
        layers.append({
            "inputs": prev_size,
            "outputs": size,
            "activation": activation
        })
        prev_size = size

    for i in range(1, n+1):
        filename = f"{base}_{i}.nn"

        weights = []
        biases = []
        prev_size = input_size

        for size in layer_sizes:
            weights.append(init_weights(prev_size, size))
            biases.append(init_biases(size))
            prev_size = size

        network = {
            "meta": {"learning_rate": learning_rate},
            "layers": layers,
            "weights": weights,
            "biases": biases,
        }

        with open(filename, "w") as f:
            json.dump(network, f, separators=(',', ':'))

        print(f"Generated {filename}")
