#include "train.hpp"
#include "fen_parser.hpp"
#include "network.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <random>

struct TrainingData {
    std::vector<double> input;
    std::vector<double> target;
};

void train_model(const AnalyzerArgs& args, json::Value& network) {
    std::ifstream file(args.data_file);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open data file: " + args.data_file);
    }
    
    std::vector<TrainingData> training_data;
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        std::vector<std::string> parts;
        std::string word;
        while (iss >> word) {
            parts.push_back(word);
        }
        
        if (parts.size() < 7) continue;
        
        // FEN (6 parts) + label
        std::string fen = parts[0] + " " + parts[1] + " " + parts[2] + " " + 
                          parts[3] + " " + parts[4] + " " + parts[5];
        
        std::string label;
        for (size_t i = 6; i < parts.size(); i++) {
            if (i > 6) label += " ";
            label += parts[i];
        }
        
        try {
            TrainingData data;
            data.input = fen_to_vector(fen);
            data.target = label_to_vector(label);
            training_data.push_back(data);
        } catch (...) {
            continue;
        }
    }
    file.close();
    
    if (training_data.empty()) {
        throw std::runtime_error("No valid training data found");
    }
    
    double base_learning_rate = network["meta"]["learning_rate"].as_number();
    
    size_t dataset_size = training_data.size();
    
    // Reduce learning rate for large datasets to prevent divergence
    double learning_rate = base_learning_rate;
    if (dataset_size > 500000) {
        learning_rate *= 0.05;  // More aggressive reduction
    } else if (dataset_size > 100000) {
        learning_rate *= 0.1;   // Changed from 0.3 to 0.1
    }
    
    double early_stop_threshold = 0.01;
    int patience = 5;
    
    int epochs = 20;
    if (dataset_size > 500000) epochs = 5;
    else if (dataset_size > 100000) epochs = 8;
    else if (dataset_size > 10000) epochs = 15;
    else if (dataset_size > 1000) epochs = 20;
    else epochs = 100;
    
    // Calculate class weights
    std::vector<double> class_counts(6, 0.0);
    for (const auto& data : training_data) {
        for (size_t i = 0; i < data.target.size(); i++) {
            if (data.target[i] == 1.0) {
                class_counts[i]++;
                break;
            }
        }
    }
    
    std::vector<double> class_weights(6);
    double total_samples = static_cast<double>(dataset_size);
    for (size_t i = 0; i < 6; i++) {
        if (class_counts[i] > 0) {
            class_weights[i] = total_samples / (6.0 * class_counts[i]);
        } else {
            class_weights[i] = 1.0;
        }
    }
    
    std::cout << "Training on " << dataset_size << " samples" << std::endl;
    std::cout << "Learning rate: " << learning_rate << " (base: " << base_learning_rate << ")" << std::endl;
    std::cout << "Class weights: [";
    for (size_t i = 0; i < class_weights.size(); i++) {
        std::cout << class_weights[i];
        if (i < class_weights.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Epochs: " << epochs << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    int no_improvement_count = 0;
    double best_loss = 1e9;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        
        std::shuffle(training_data.begin(), training_data.end(), gen);
        
        // Adaptive learning rate: reduce by half each epoch after epoch 1 if loss is high
        double current_lr = learning_rate;
        if (epoch > 0 && best_loss > 5.0) {
            current_lr = learning_rate * std::pow(0.95, epoch);
        }
        
        for (size_t i = 0; i < training_data.size(); i++) {
            ForwardCache cache;
            auto output = forward_pass(network, training_data[i].input, cache);
            double loss = cross_entropy_loss(output, training_data[i].target, class_weights);
            total_loss += loss;
            
            // Skip updates with extreme loss to prevent divergence
            if (loss > 10.0) continue;
            
            backward_pass(network, cache, training_data[i].target, current_lr, true);
        }
        
        double avg_loss = total_loss / training_data.size();
        
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                  << ", Loss: " << avg_loss << " (lr: " << current_lr << ")" << std::endl;
        
        // Early stopping
        if (avg_loss < early_stop_threshold) {
            std::cout << "Loss below threshold (" << early_stop_threshold << "), stopping early!" << std::endl;
            break;
        }
        
        if (avg_loss < best_loss) {
            best_loss = avg_loss;
            no_improvement_count = 0;
        } else {
            no_improvement_count++;
            if (no_improvement_count >= patience) {
                std::cout << "No improvement for " << patience << " epochs, stopping early!" << std::endl;
                break;
            }
        }
    }
    
    // Save
    std::ofstream out(args.save_file);
    out << json::stringify(network, false);
    out.close();
    
    std::cout << "Training complete. Network saved to " << args.save_file << std::endl;
}
