#include "generator.hpp"
#include "../include/json_parser.hpp"
#include <fstream>
#include <cmath>
#include <random>
#include <iostream>

static std::vector<std::vector<double>> init_weights(int input_size, int output_size) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    double limit = std::sqrt(6.0 / (input_size + output_size));
    std::uniform_real_distribution<> dis(-limit, limit);
    
    std::vector<std::vector<double>> weights(output_size, std::vector<double>(input_size));
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            weights[i][j] = dis(gen);
        }
    }
    return weights;
}

static std::vector<double> init_biases(int output_size) {
    return std::vector<double>(output_size, 0.0);
}

void generate_network(const std::string& config_file, const NetworkConfig& config, int n) {
    std::string base = config_file;
    size_t dot_pos = base.rfind(".conf");
    if (dot_pos != std::string::npos) {
        base = base.substr(0, dot_pos);
    }
    
    for (int i = 1; i <= n; i++) {
        std::string filename = base + "_" + std::to_string(i) + ".nn";
        
        json::Value network;
        network.set_object({});
        
        // Meta
        json::Value meta;
        meta.set_object({});
        meta["learning_rate"] = json::Value(config.learning_rate);
        network["meta"] = meta;
        
        // Layers
        json::Value layers;
        std::vector<json::Value> layers_arr;
        int prev_size = config.input_size;
        for (size_t j = 0; j < config.layer_sizes.size(); j++) {
            json::Value layer;
            layer.set_object({});
            layer["inputs"] = json::Value(prev_size);
            layer["outputs"] = json::Value(config.layer_sizes[j]);
            layer["activation"] = json::Value(config.activations[j]);
            layers_arr.push_back(layer);
            prev_size = config.layer_sizes[j];
        }
        layers.set_array(layers_arr);
        network["layers"] = layers;
        
        // Weights
        json::Value weights;
        std::vector<json::Value> weights_arr;
        prev_size = config.input_size;
        for (int size : config.layer_sizes) {
            auto w = init_weights(prev_size, size);
            std::vector<json::Value> w_layer;
            for (const auto& row : w) {
                std::vector<json::Value> w_row;
                for (double val : row) {
                    w_row.push_back(json::Value(val));
                }
                json::Value row_val;
                row_val.set_array(w_row);
                w_layer.push_back(row_val);
            }
            json::Value layer_val;
            layer_val.set_array(w_layer);
            weights_arr.push_back(layer_val);
            prev_size = size;
        }
        weights.set_array(weights_arr);
        network["weights"] = weights;
        
        // Biases
        json::Value biases;
        std::vector<json::Value> biases_arr;
        for (int size : config.layer_sizes) {
            auto b = init_biases(size);
            std::vector<json::Value> b_layer;
            for (double val : b) {
                b_layer.push_back(json::Value(val));
            }
            json::Value layer_val;
            layer_val.set_array(b_layer);
            biases_arr.push_back(layer_val);
        }
        biases.set_array(biases_arr);
        network["biases"] = biases;
        
        // Save to file
        std::ofstream file(filename);
        file << json::stringify(network, true);
        file.close();
        
        std::cout << "Generated " << filename << std::endl;
    }
}
