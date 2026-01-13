#include "parsor.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <cstdlib>

static std::string trim(const std::string& s) {
    size_t start = 0, end = s.size();
    while (start < end && std::isspace(s[start])) start++;
    while (end > start && std::isspace(s[end - 1])) end--;
    return s.substr(start, end - start);
}

static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        std::string trimmed = trim(item);
        if (!trimmed.empty()) result.push_back(trimmed);
    }
    return result;
}

NetworkConfig parse_config_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + path);
    }
    
    std::map<std::string, std::string> raw_config;
    std::string line;
    
    while (std::getline(file, line)) {
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        line = trim(line);
        if (line.empty()) continue;
        
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        
        std::string key = trim(line.substr(0, eq_pos));
        std::string val = trim(line.substr(eq_pos + 1));
        raw_config[key] = val;
    }
    
    NetworkConfig config;
    
    if (raw_config.find("input_size") != raw_config.end()) {
        config.input_size = std::stoi(raw_config["input_size"]);
    } else {
        throw std::runtime_error("Missing input_size in config");
    }
    
    if (raw_config.find("layer_sizes") != raw_config.end()) {
        auto parts = split(raw_config["layer_sizes"], ',');
        for (const auto& p : parts) {
            config.layer_sizes.push_back(std::stoi(p));
        }
    } else {
        throw std::runtime_error("Missing layer_sizes in config");
    }
    
    if (raw_config.find("activations") != raw_config.end()) {
        config.activations = split(raw_config["activations"], ',');
    } else {
        throw std::runtime_error("Missing activations in config");
    }
    
    if (raw_config.find("learning_rate") != raw_config.end()) {
        config.learning_rate = std::stod(raw_config["learning_rate"]);
    } else {
        config.learning_rate = 0.01;
    }
    
    return config;
}

std::vector<std::pair<std::string, int>> parse_cli_arguments(int argc, char* argv[]) {
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << "USAGE\n"
                  << "    ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]\n\n"
                  << "DESCRIPTION\n"
                  << "    config_file_i    Configuration file describing the neural network.\n"
                  << "    nb_i             Number of networks to generate from this config.\n";
        std::exit(0);
    }
    
    if (argc < 3 || (argc - 1) % 2 != 0) {
        throw std::runtime_error("Invalid number of arguments");
    }
    
    std::vector<std::pair<std::string, int>> result;
    for (int i = 1; i < argc; i += 2) {
        std::string config_file = argv[i];
        int nb = std::atoi(argv[i + 1]);
        if (nb <= 0) {
            throw std::runtime_error("Number of network must be > 0");
        }
        result.push_back({config_file, nb});
    }
    
    return result;
}
