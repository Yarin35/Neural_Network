CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -O3 -I./include
LDFLAGS = -lm

GENERATOR_SRCS = generator_cpp/main.cpp generator_cpp/parsor.cpp generator_cpp/generator.cpp include/json_parser.cpp
ANALYZER_SRCS = analyzer_cpp/main.cpp analyzer_cpp/parsor.cpp analyzer_cpp/fen_parser.cpp analyzer_cpp/network.cpp analyzer_cpp/train.cpp analyzer_cpp/predict.cpp include/json_parser.cpp

GENERATOR_BIN = my_torch_generator
ANALYZER_BIN = my_torch_analyzer

all: $(GENERATOR_BIN) $(ANALYZER_BIN)

$(GENERATOR_BIN): $(GENERATOR_SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(ANALYZER_BIN): $(ANALYZER_SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f *.o generator_cpp/*.o analyzer_cpp/*.o include/*.o

fclean: clean
	rm -f $(GENERATOR_BIN) $(ANALYZER_BIN)

re: fclean all

.PHONY: all clean fclean re