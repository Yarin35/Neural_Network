#include "parsor.hpp"
#include <iostream>
#include <stdexcept>
#include <cstdlib>

AnalyzerArgs parse_analyzer_arguments(int argc, char* argv[]) {
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << "USAGE\n"
                  << "    ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE FILE\n\n"
                  << "DESCRIPTION\n"
                  << "    --train     Launch in training mode. FILE contains FEN positions and labels.\n"
                  << "    --predict   Launch in prediction mode. FILE contains FEN positions.\n"
                  << "    --save      Save network to SAVEFILE (train mode only).\n"
                  << "    LOADFILE    File containing the neural network.\n"
                  << "    FILE        File containing chessboards in FEN notation.\n";
        std::exit(0);
    }
    
    AnalyzerArgs args;
    args.mode = "";
    args.load_file = "";
    args.data_file = "";
    args.save_file = "";
    args.debug_mode = false;
    
    int i = 1;
    while (i < argc) {
        std::string arg = argv[i];
        
        if (arg == "--train") {
            args.mode = "train";
        } else if (arg == "--predict") {
            args.mode = "predict";
        } else if (arg == "--save") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--save requires a filename");
            }
            args.save_file = argv[i + 1];
            i++;
        } else if (args.load_file.empty()) {
            args.load_file = arg;
        } else if (args.data_file.empty()) {
            args.data_file = arg;
        } else if (arg == "--mode=debug") {
            args.debug_mode = true;
        }
        i++;
    }
    
    if (args.mode.empty() || args.load_file.empty() || args.data_file.empty()) {
        throw std::runtime_error("Missing required arguments");
    }
    
    if (args.save_file.empty()) {
        args.save_file = args.load_file;
    }
    
    return args;
}
