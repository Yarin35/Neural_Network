#include "fen_parser.hpp"
#include <sstream>
#include <stdexcept>
#include <map>
#include <algorithm>
#include <cctype>

static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

std::vector<double> fen_to_vector(const std::string& fen) {
    std::istringstream iss(fen);
    std::string board_part, turn;
    iss >> board_part >> turn;
    
    if (board_part.empty() || turn.empty()) {
        throw std::runtime_error("Invalid FEN: " + fen);
    }
    
    std::map<char, int> piece_map = {
        {'P', 0}, {'N', 1}, {'B', 2}, {'R', 3}, {'Q', 4}, {'K', 5},
        {'p', 6}, {'n', 7}, {'b', 8}, {'r', 9}, {'q', 10}, {'k', 11}
    };
    
    std::vector<double> vec(769, 0.0);
    
    int square = 0;
    for (char c : board_part) {
        if (c == '/') {
            continue;
        } else if (std::isdigit(c)) {
            square += (c - '0');
        } else {
            auto it = piece_map.find(c);
            if (it != piece_map.end()) {
                vec[square * 12 + it->second] = 1.0;
            }
            square++;
        }
    }
    
    vec[768] = (turn == "w" || turn == "W") ? 1.0 : 0.0;
    
    return vec;
}

std::vector<double> label_to_vector(const std::string& label) {
    std::string lower_label = to_lower(label);
    
    std::map<std::string, int> labels = {
        {"nothing", 0},
        {"check white", 1},
        {"check black", 2},
        {"checkmate white", 3},
        {"checkmate black", 4},
        {"stalemate", 5}
    };
    
    auto it = labels.find(lower_label);
    if (it == labels.end()) {
        throw std::runtime_error("Invalid label: " + label);
    }
    
    std::vector<double> vec(6, 0.0);
    vec[it->second] = 1.0;
    return vec;
}

std::string vector_to_label(const std::vector<double>& vec) {
    std::vector<std::string> labels = {"Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black", "Stalemate"};
    int max_idx = 0;
    for (size_t i = 1; i < vec.size(); i++) {
        if (vec[i] > vec[max_idx]) {
            max_idx = i;
        }
    }
    return labels[max_idx];
}
