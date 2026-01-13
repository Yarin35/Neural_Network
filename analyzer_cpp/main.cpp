#include "parsor.hpp"
#include "train.hpp"
#include "predict.hpp"
#include "../include/json_parser.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

int main(int argc, char* argv[]) {
    try {
        AnalyzerArgs args = parse_analyzer_arguments(argc, argv);
        
        // Load network
        std::ifstream file(args.load_file);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open network file: " + args.load_file);
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        file.close();
        
        json::Value network = json::parse(buffer.str());
        
        if (args.mode == "train") {
            train_model(args, network);
        } else if (args.mode == "predict") {
            predict_model(args, network);
        } else {
            throw std::runtime_error("Invalid mode specified. Use --train or --predict.");
        }
        
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 84;
    }
    
    return 0;
}
