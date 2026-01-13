#pragma once
#include <string>
#include <map>

struct AnalyzerArgs {
    std::string mode;
    std::string load_file;
    std::string data_file;
    std::string save_file;
    bool debug_mode;
};

AnalyzerArgs parse_analyzer_arguments(int argc, char* argv[]);
