#include "network.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

static double relu(double x) {
    return std::max(0.0, x);
}

static double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

static std::vector<double> softmax(const std::vector<double>& vec) {
    double max_val = *std::max_element(vec.begin(), vec.end());
    std::vector<double> exps;
    double sum = 0.0;
    
    for (double x : vec) {
        double e = std::exp(std::min(x - max_val, 700.0));
        exps.push_back(e);
        sum += e;
    }
    
    if (sum < 1e-10) sum = 1e-10;
    
    for (auto& e : exps) {
        e /= sum;
    }
    
    return exps;
}

static std::vector<double> activate(const std::vector<double>& vec, const std::string& activation) {
    if (activation == "relu") {
        std::vector<double> result;
        for (double x : vec) {
            result.push_back(relu(x));
        }
        return result;
    } else if (activation == "softmax") {
        return softmax(vec);
    }
    return vec;
}

static std::vector<double> activate_derivative(const std::vector<double>& vec, const std::string& activation) {
    if (activation == "relu") {
        std::vector<double> result;
        for (double x : vec) {
            result.push_back(relu_derivative(x));
        }
        return result;
    }
    return std::vector<double>(vec.size(), 1.0);
}

static std::vector<double> matrix_vector_mult(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vec) {
    std::vector<double> result(matrix.size(), 0.0);
    for (size_t i = 0; i < matrix.size(); i++) {
        double sum = 0.0;
        for (size_t j = 0; j < vec.size(); j++) {
            sum += matrix[i][j] * vec[j];
        }
        result[i] = sum;
    }
    return result;
}

static std::vector<double> vector_add(const std::vector<double>& v1, const std::vector<double>& v2) {
    std::vector<double> result(v1.size());
    for (size_t i = 0; i < v1.size(); i++) {
        result[i] = v1[i] + v2[i];
    }
    return result;
}

std::vector<double> forward_pass(json::Value& network, const std::vector<double>& input, ForwardCache& cache) {
    cache.activations.clear();
    cache.z_values.clear();
    cache.activations.push_back(input);
    
    std::vector<double> current = input;
    
    const auto& layers = network["layers"].as_array();
    const auto& weights_arr = network["weights"].as_array();
    const auto& biases_arr = network["biases"].as_array();
    
    for (size_t i = 0; i < layers.size(); i++) {
        const auto& layer = layers[i];
        std::string activation = layer["activation"].as_string();
        
        // Extract weights
        std::vector<std::vector<double>> weights;
        const auto& w_layer = weights_arr[i].as_array();
        for (const auto& row : w_layer) {
            std::vector<double> w_row;
            for (const auto& val : row.as_array()) {
                w_row.push_back(val.as_number());
            }
            weights.push_back(w_row);
        }
        
        // Extract biases
        std::vector<double> biases;
        for (const auto& val : biases_arr[i].as_array()) {
            biases.push_back(val.as_number());
        }
        
        // Compute z = W*x + b
        std::vector<double> z = vector_add(matrix_vector_mult(weights, current), biases);
        cache.z_values.push_back(z);
        
        // Activation
        current = activate(z, activation);
        cache.activations.push_back(current);
    }
    
    return current;
}

double cross_entropy_loss(const std::vector<double>& predicted, const std::vector<double>& target, const std::vector<double>& class_weights) {
    const double epsilon = 1e-15;
    double loss = 0.0;
    
    // Find which class is active (one-hot encoded)
    double weight = 1.0;
    if (!class_weights.empty()) {
        for (size_t i = 0; i < target.size(); i++) {
            if (target[i] == 1.0) {
                weight = class_weights[i];
                break;
            }
        }
    }
    
    for (size_t i = 0; i < target.size(); i++) {
        double p = std::max(epsilon, std::min(1.0 - epsilon, predicted[i]));
        loss -= target[i] * std::log(p) * weight;
    }
    
    return loss;
}

Gradients backward_pass(json::Value& network, const ForwardCache& cache, const std::vector<double>& target, double learning_rate, bool apply_update) {
    const auto& layers = network["layers"].as_array();
    auto& weights_arr = network["weights"];
    auto& biases_arr = network["biases"];
    
    const auto& activations = cache.activations;
    const auto& z_values = cache.z_values;
    
    size_t num_layers = layers.size();
    
    // Output gradient
    std::vector<double> delta;
    const auto& output = activations.back();
    for (size_t i = 0; i < output.size(); i++) {
        delta.push_back(output[i] - target[i]);
    }
    
    Gradients grads;
    
    // Backpropagate
    for (int i = num_layers - 1; i >= 0; i--) {
        const auto& prev_activation = activations[i];
        
        // Get weights/biases
        auto& w_layer = weights_arr[i];
        auto& b_layer = biases_arr[i];
        size_t output_size = w_layer.size();
        size_t input_size = w_layer[0].size();
        
        // Compute gradients
        std::vector<std::vector<double>> w_grad(output_size, std::vector<double>(input_size));
        std::vector<double> b_grad(output_size);
        
        const double grad_clip = 5.0;
        
        for (size_t j = 0; j < output_size; j++) {
            for (size_t k = 0; k < input_size; k++) {
                double grad = delta[j] * prev_activation[k];
                w_grad[j][k] = std::max(-grad_clip, std::min(grad_clip, grad));
            }
            b_grad[j] = std::max(-grad_clip, std::min(grad_clip, delta[j]));
        }
        
        grads.weights.insert(grads.weights.begin(), w_grad);
        grads.biases.insert(grads.biases.begin(), b_grad);
        
        // Apply updates
        if (apply_update) {
            for (size_t j = 0; j < output_size; j++) {
                for (size_t k = 0; k < input_size; k++) {
                    double old_val = w_layer[j][k].as_number();
                    w_layer[j][k] = json::Value(old_val - learning_rate * w_grad[j][k]);
                }
                double old_bias = b_layer[j].as_number();
                b_layer[j] = json::Value(old_bias - learning_rate * b_grad[j]);
            }
        }
        
        // Propagate error
        if (i > 0) {
            std::vector<double> next_delta(prev_activation.size(), 0.0);
            
            for (size_t j = 0; j < prev_activation.size(); j++) {
                double error = 0.0;
                for (size_t k = 0; k < delta.size(); k++) {
                    error += w_layer[k][j].as_number() * delta[k];
                }
                
                std::string prev_activation_fn = layers[i-1]["activation"].as_string();
                auto deriv = activate_derivative(z_values[i-1], prev_activation_fn);
                next_delta[j] = error * deriv[j];
            }
            
            delta = next_delta;
        }
    }
    
    return grads;
}

void accumulate_gradients(Gradients& g1, const Gradients& g2) {
    if (g1.weights.empty()) {
        g1 = g2;
        return;
    }
    
    for (size_t i = 0; i < g1.weights.size(); i++) {
        for (size_t j = 0; j < g1.weights[i].size(); j++) {
            for (size_t k = 0; k < g1.weights[i][j].size(); k++) {
                g1.weights[i][j][k] += g2.weights[i][j][k];
            }
        }
    }
    
    for (size_t i = 0; i < g1.biases.size(); i++) {
        for (size_t j = 0; j < g1.biases[i].size(); j++) {
            g1.biases[i][j] += g2.biases[i][j];
        }
    }
}

void scale_gradients(Gradients& grads, double scale) {
    for (auto& w_layer : grads.weights) {
        for (auto& row : w_layer) {
            for (auto& val : row) {
                val *= scale;
            }
        }
    }
    
    for (auto& b_layer : grads.biases) {
        for (auto& val : b_layer) {
            val *= scale;
        }
    }
}

void apply_gradients(json::Value& network, const Gradients& grads, double learning_rate) {
    auto& weights_arr = network["weights"];
    auto& biases_arr = network["biases"];
    
    for (size_t i = 0; i < grads.weights.size(); i++) {
        auto& w_layer = weights_arr[i];
        for (size_t j = 0; j < grads.weights[i].size(); j++) {
            for (size_t k = 0; k < grads.weights[i][j].size(); k++) {
                double old_val = w_layer[j][k].as_number();
                w_layer[j][k] = json::Value(old_val - learning_rate * grads.weights[i][j][k]);
            }
        }
    }
    
    for (size_t i = 0; i < grads.biases.size(); i++) {
        auto& b_layer = biases_arr[i];
        for (size_t j = 0; j < grads.biases[i].size(); j++) {
            double old_val = b_layer[j].as_number();
            b_layer[j] = json::Value(old_val - learning_rate * grads.biases[i][j]);
        }
    }
}
