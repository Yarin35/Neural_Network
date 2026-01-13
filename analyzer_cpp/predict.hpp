#pragma once
#include "parsor.hpp"
#include "../include/json_parser.hpp"

void predict_model(const AnalyzerArgs& args, json::Value& network);
