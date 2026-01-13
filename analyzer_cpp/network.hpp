#pragma once
#include "../include/json_parser.hpp"
#include <vector>

struct ForwardCache {
    std::vector<std::vector<double>> activations;
    std::vector<std::vector<double>> z_values;
};

struct Gradients {
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
};

std::vector<double> forward_pass(json::Value& network, const std::vector<double>& input, ForwardCache& cache);
double cross_entropy_loss(const std::vector<double>& predicted, const std::vector<double>& target, const std::vector<double>& class_weights = {});
Gradients backward_pass(json::Value& network, const ForwardCache& cache, const std::vector<double>& target, double learning_rate, bool apply_update);
void accumulate_gradients(Gradients& g1, const Gradients& g2);
void scale_gradients(Gradients& grads, double scale);
void apply_gradients(json::Value& network, const Gradients& grads, double learning_rate);
