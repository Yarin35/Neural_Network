#include "predict.hpp"
#include "fen_parser.hpp"
#include "network.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

void predict_model(const AnalyzerArgs& args, json::Value& network) {
    std::ifstream file(args.data_file);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open data file: " + args.data_file);
    }
    
    int total = 0;
    int correct = 0;
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        std::vector<std::string> parts;
        std::string word;
        while (iss >> word) {
            parts.push_back(word);
        }
        
        std::string fen;
        std::string expected;
        bool has_expected = false;
        
        if (parts.size() >= 7) {
            fen = parts[0] + " " + parts[1] + " " + parts[2] + " " + 
                  parts[3] + " " + parts[4] + " " + parts[5];
            
            for (size_t i = 6; i < parts.size(); i++) {
                if (i > 6) expected += " ";
                expected += parts[i];
            }
            has_expected = true;
        } else {
            fen = line;
        }
        
        try {
            auto input = fen_to_vector(fen);
            ForwardCache cache;
            auto output = forward_pass(network, input, cache);
            std::string prediction = vector_to_label(output);
            
            // Add color for Check/Checkmate
            if (prediction == "Check" || prediction == "Checkmate") {
                std::istringstream fen_iss(fen);
                std::string board, turn;
                fen_iss >> board >> turn;
                if (!turn.empty()) {
                    std::string color = (turn == "W" || turn == "w") ? "White" : "Black";
                    prediction = prediction + " " + color;
                }
            }
            
            if (args.debug_mode && has_expected) {
                total++;
                bool is_correct = (prediction == expected);
                if (is_correct) correct++;
                
                std::string status = is_correct ? "✓" : "✗";
                std::cout << status << " " << prediction << " (expected: " << expected << ")" << std::endl;
            } else {
                std::cout << prediction << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error processing FEN: " << e.what() << std::endl;
        }
    }
    
    file.close();
    
    if (args.debug_mode && total > 0) {
        double accuracy = (double)correct / total * 100.0;
        std::cout << "\n==================================================\n";
        std::cout << "Results: " << correct << "/" << total << " correct (" << accuracy << "%)\n";
        std::cout << "==================================================\n";
    }
}
