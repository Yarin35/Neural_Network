#include "parsor.hpp"
#include "generator.hpp"
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    try {
        auto config_pairs = parse_cli_arguments(argc, argv);
        
        for (const auto& pair : config_pairs) {
            auto config = parse_config_file(pair.first);
            generate_network(pair.first, config, pair.second);
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 84;
    }
    
    return 0;
}
