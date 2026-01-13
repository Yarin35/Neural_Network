#pragma once
#include <string>
#include <vector>
#include <map>

struct NetworkConfig {
    int input_size;
    std::vector<int> layer_sizes;
    std::vector<std::string> activations;
    double learning_rate;
};

NetworkConfig parse_config_file(const std::string& path);
std::vector<std::pair<std::string, int>> parse_cli_arguments(int argc, char* argv[]);
