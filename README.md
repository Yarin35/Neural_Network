# My Torch - Neural Network Framework

## Project Overview

**My Torch** is a from-scratch neural network implementation in **C++** for chess position analysis. The system classifies chess positions (FEN notation) into 6 categories:

- **Nothing White/Black**: Normal position (white's or black's perspective)
- **Check White/Black**: King is in check (indicating which side has advantage)
- **Checkmate White/Black**: Game over (indicating winner)

**Key Constraint**: Pure C++ implementation - NO external lib too useful allowed (keep from scratch approach).

---

## Architecture

### Network Topology

```bash
Input Layer (769 neurons)
    ↓
Hidden Layer 1 (128 neurons, ReLU)
    ↓
Hidden Layer 2 (64 neurons, ReLU)
    ↓
Output Layer (6 neurons, Softmax)
```

### Input Encoding (769 dimensions)

- **768 neurons**: Board state (64 squares × 12 piece types, one-hot)
  - White pieces: P=0, N=1, B=2, R=3, Q=4, K=5
  - Black pieces: p=6, n=7, b=8, r=9, q=10, k=11
- **1 neuron**: Turn indicator (white=1.0, black=0.0)

### Output Classes

| Class           | Index | One-Hot Vector     |
| --------------- | ----- | ------------------ |
| Nothing White   | 0     | [1, 0, 0, 0, 0, 0] |
| Nothing Black   | 1     | [0, 1, 0, 0, 0, 0] |
| Check White     | 2     | [0, 0, 1, 0, 0, 0] |
| Check Black     | 3     | [0, 0, 0, 1, 0, 0] |
| Checkmate White | 4     | [0, 0, 0, 0, 1, 0] |
| Checkmate Black | 5     | [0, 0, 0, 0, 0, 1] |

---

## Usage

### 1. Generate a Network

```bash
./my_torch_generator network.conf 1
```

**Configuration file** (`network.conf`):

```ini
input_size=769
layer_sizes=128,64,6
activations=relu,relu,softmax
learning_rate=0.001
```

Output: `network_1.nn` (JSON format, ~1.1MB)

### 2. Train the Network

```bash
./my_torch_analyzer --train network_1.nn training_data.txt --save my_torch_network.nn
```

**Training data format**:

```bash
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 nothing
r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4 checkmate White
rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3 checkmate Black
```

### 3. Make Predictions

```bash
./my_torch_analyzer --predict my_torch_network.nn test_positions.txt
```

Output:

```bash
Nothing White (expected: nothing)
Checkmate White (expected: checkmate White)
Check Black (expected: check Black)
```

---

## Benchmarks & Results

### Training Configuration

- **Implementation**: C++ from scratch (no external ML libraries)
- **Dataset**: Balanced dataset with all 6 classes
  - Nothing White/Black positions (both are the same result => Nothing)
  - Check White/Black positions
  - Checkmate White/Black positions
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Learning Rate**: 0.001
- **Epochs**: Multiple training sessions
- **Batch Size**: 1 (online learning)
- **Loss Function**: Cross-Entropy

### Performance Metrics

| Metric             | Value                                |
| ------------------ | ------------------------------------ |
| **Accuracy Range** | **68-78%** (varies per testing data) |
| Network Size       | 1.1 MB (JSON format)                 |
| Parameters         | ~106,694 (769×128 + 128×64 + 64×6)   |
| Training Time      | Varies by dataset size               |
| Implementation     | Pure C++                             |

### Accuracy Analysis

**Current Performance**: 68-78% accuracy on test datasets

**Strengths**:

- Good recognition of basic positions (Nothing)
- Effective checkmate detection in clear cases
- Fast inference time thanks to C++ implementation

**Areas for Improvement**:

- Check vs Checkmate distinction needs refinement
- Edge cases with complex positions
- Balance between white/black perspective predictions

**Variance**: The 10% accuracy range (68-78%) is due to:

- Random weight initialization (Xavier/Glorot method)
- Stochastic gradient descent randomness
- Training data shuffling order

## Mathematical Foundation

### Forward Propagation

For each layer `l`:

```bash
z^[l] = W^[l] · a^[l-1] + b^[l]
a^[l] = g(z^[l])
```

### Activation Functions

**ReLU** (Hidden Layers):

```bash
f(x) = max(0, x)
f'(x) = 1 if x > 0 else 0
```

**Softmax** (Output Layer):

```bash
f(x_i) = exp(x_i) / Σ exp(x_j)
```

### Loss Function

**Cross-Entropy**:

```bash
L = -Σ y_i · log(ŷ_i)
```

### Backpropagation

Output layer gradient:

```bash
δ^[L] = a^[L] - y
```

Hidden layer gradients:

```bash
δ^[l] = (W^[l+1])^T · δ^[l+1] ⊙ g'(z^[l])
```

Weight updates:

```bash
W^[l] := W^[l] - α · δ^[l] · (a^[l-1])^T
b^[l] := b^[l] - α · δ^[l]
```

### Weight Initialization

**Xavier/Glorot** method:

```python
limit = sqrt(6 / (input_size + output_size))
W ~ U[-limit, +limit]
```

Ensures stable variance across layers.

---

## Project Structure

```bash
.
├── my_torch_generator          # Binary (generator)
├── my_torch_analyzer           # Binary (analyzer)
├── my_torch_network.nn         # Pre-trained network (on all training dataset)
├── network.conf                # Network configuration
├── Makefile                    # Build system
├── data/
│   ├── test_data.txt
│   ├── small_dataset.txt
│   ├── large_dataset.txt
│   ├── max_balanced_dataset.txt
│   ├── dataset/
│   │   ├── test_light.txt  # small testing file (100 lines)
│   │   ├── test_medium.txt # medium size testing file (1 000 lines)
│   │   └── test_heavy.txt  # large testing file (10 000 lines)
│   └── dataset/
│       ├── check/              # Check positions
│       ├── nothing/            # Normal positions
│       └── checkmate/          # Checkmate positions
├── generator_cpp/
│   ├── main.cpp                # Generator entry point
│   ├── parsor.cpp              # Config file parser
│   └── generator.cpp           # Network initialization
├── analyzer_cpp/
│   ├── main.cpp                # Analyzer entry point
│   ├── parsor.cpp              # Argument parser
│   ├── fen_parser.cpp          # FEN to neural input
│   ├── network.cpp             # Forward/backward pass
│   ├── train.cpp               # Training logic
│   └── predict.cpp             # Prediction logic
└── include/
    ├── json_parser.cpp         # JSON serialization
    └── json_parser.hpp         # JSON header
```

---

## Building

```bash
make  # to build
make clean  # to clean 
make fclean # clean advanced
make re # to clean and build
```

---

## Design Choices & Justifications

### 1. Why 769 input neurons?

- **64 squares × 12 piece types** = 768 (one-hot encoding)
- **+1 for turn**: Critical for positions where only the turn differentiates states
- Captures complete board state in a format suitable for neural networks

### 2. Why 128 → 64 hidden layers?

- **Gradual compression**: 769 → 128 → 64 → 6 provides smooth feature extraction
- **Tested alternatives**:
  - Single layer (256): Underfitting, poor accuracy
  - Three layers (256→128→64): Overfitting on smaller datasets
- **Trade-off**: Complexity vs. training time vs. generalization

### 3. Why ReLU activation?

- **Computational efficiency**: Simple max(0, x) - very fast in C++
- **No vanishing gradients**: Unlike sigmoid/tanh
- **Industry standard**: Proven for deep networks
- Easy derivative: f'(x) = 1 if x > 0 else 0

### 4. Why learning rate = 0.001?

- **Initial attempts** (0.01, 0.005): Caused instability or NaN values
- **Final choice** (0.001): Stable convergence, smooth loss decrease
- **Trade-off**: Slower training but more reliable convergence
- **Future improvement**: Adaptive learning rate (Adam) could speed up training

### 5. Why Cross-Entropy loss?

- **Multi-class classification**: Natural fit for 6 output classes
- **Probabilistic interpretation**: Softmax + Cross-Entropy = clean gradients
- **Mathematical elegance**: ∂L/∂z = ŷ - y (simple gradient)
- **Alternative**: MSE works but slower convergence and less suitable for classification

### 6. Why C++ implementation?

- **Performance**: 10-100x faster than Python for matrix operations
- **No dependencies**: Self-contained, no external ML libraries required
- **Educational value**: Complete understanding of every algorithm
- **Project constraint**: Must be from-scratch implementation
- **Deployment**: Single binary, easy distribution

### 7. Why 6 output classes instead of 3?

- **Finer granularity**: Distinguishes white advantage vs black advantage
- **Better training**: Network learns perspective-aware features
- **More informative**: Tells not just the state but who benefits
- **Matches chess semantics**: "Check White" means white has the advantage

---

## Sources

- Glorot, X., & Bengio, Y. (2010). _Understanding the difficulty of training deep feedforward neural networks_
- Nielsen, M. (2015). _Neural Networks and Deep Learning_
- FEN Notation: [Wikipedia](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation)
