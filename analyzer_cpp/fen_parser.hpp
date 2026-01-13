#pragma once
#include <vector>
#include <string>

std::vector<double> fen_to_vector(const std::string& fen);
std::vector<double> label_to_vector(const std::string& label);
std::string vector_to_label(const std::vector<double>& vec);
