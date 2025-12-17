# My Torch - Neural Network Framework

## Project Overview

**My Torch** is a from-scratch neural network implementation in Python for chess position analysis. The system classifies chess positions (FEN notation) into 4 categories:

- **Nothing**: Normal position
- **Check**: King is in check
- **Checkmate**: Game over
- **Stalemate**: Draw position

**Key Constraint**: Pure Python implementation - NO PyTorch/TensorFlow allowed.

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
Output Layer (4 neurons, Softmax)
```

### Input Encoding (769 dimensions)

- **768 neurons**: Board state (64 squares × 12 piece types, one-hot)
  - White pieces: P=0, N=1, B=2, R=3, Q=4, K=5
  - Black pieces: p=6, n=7, b=8, r=9, q=10, k=11
- **1 neuron**: Turn indicator (white=1.0, black=0.0)

### Output Classes

| Class     | Index | One-Hot Vector |
| --------- | ----- | -------------- |
| Nothing   | 0     | [1, 0, 0, 0]   |
| Check     | 1     | [0, 1, 0, 0]   |
| Checkmate | 2     | [0, 0, 1, 0]   |
| Stalemate | 3     | [0, 0, 0, 1]   |

---

## Usage

### 1. Generate a Network

```bash
./my_torch_generator network.conf 1
```

**Configuration file** (`network.conf`):

```ini
input_size=769
layer_sizes=128,64,4
activations=relu,relu,softmax
learning_rate=0.001
```

Output: `network_1.nn` (JSON format, ~4MB)

### 2. Train the Network

```bash
./my_torch_analyzer --train network_1.nn training_data.txt --save my_torch_network.nn
```

**Training data format**:

```bash
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 nothing
r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4 checkmate
```

### 3. Make Predictions

```bash
./my_torch_analyzer --predict my_torch_network.nn test_positions.txt
```

Output:

```bash
Nothing (expected: nothing)
Checkmate (expected: checkmate)
```

---

## Benchmarks & Results

### Training Configuration

- **Dataset**: 2000 samples (from 473K positions)
  - 1000 Nothing positions
  - 800 Checkmate positions
  - no check positions
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Learning Rate**: 0.001
- **Epochs**: 20
- **Batch Size**: 1 (online learning)
- **Loss Function**: Cross-Entropy

### Training Results

| Epoch | Loss   |
| ----- | ------ |
| 0     | 0.9447 |
| 2     | 0.6551 |
| 4     | 0.6166 |
| 6     | 0.5794 |
| 8     | 0.5357 |
| 10    | 0.4849 |
| 12    | 0.4300 |
| 14    | 0.3748 |
| 16    | 0.3219 |
| 18    | 0.2719 |

> result based on latest test might be slightly different between each run but should be globally the same.

> The lack of check position is significant in the lack of check guess from the network

**Convergence**: Smooth decrease from 0.94 → 0.27 (71% reduction)

### Performance Metrics

| Metric         | Value                                |
| -------------- | ------------------------------------ |
| Training Time  | ~8 minutes (2000 samples, 20 epochs) |
| Network Size   | 3.9 MB                               |
| Inference Time | < 0.01s per position                 |
| Parameters     | ~98,820 (769×128 + 128×64 + 64×4)    |

### Sample Predictions

| Position             | Ground Truth | Prediction | ✓/✗ |
| -------------------- | ------------ | ---------- | --- |
| Starting position    | Nothing      | Nothing    | ✓   |
| Fool's mate          | Checkmate    | Nothing    | ✗   |
| King vs King         | Stalemate    | Checkmate  | ✗   |
| Scholar's mate setup | Check        | Checkmate  | ✗   |

**Observed Issue**: Model needs more training data for Check/Stalemate classes (underrepresented).

---

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
├── my_torch_generator
├── my_torch_analyzer
├── my_torch_network.nn
├── network.conf
├── dataset/
│   ├── check/
│   ├── nothing/
│   └── checkmate/
├── generator/
│   ├── main.py
│   ├── parsor.py
│   └── generator.py
├── analyzer/
│   ├── main.py
│   ├── parsor.py
│   ├── fen_parser.py
│   ├── network.py
│   ├── train.py
│   └── predict.py
└── makefile
```

---

## Building

```bash
make
make clean
make fclean
make re
```

---

## Design Choices & Justifications

### 1. Why 769 input neurons?

- **64 squares × 12 piece types** = 768 (one-hot encoding)
- **+1 for turn**: Critical for positions where only the turn differentiates states

### 2. Why 128 → 64 hidden layers?

- **Gradual compression**: 769 → 128 → 64 → 4 provides smooth feature extraction
- **Tested alternatives**:
  - Single layer (256): Underfitting
  - Three layers (256→128→64): Overfitting on small dataset
- **Trade-off**: Complexity vs. training time

### 3. Why ReLU activation?

- **Computational efficiency**: Simple max(0, x)
- **No vanishing gradients**: Unlike sigmoid/tanh
- **Industry standard**: Proven for deep networks

### 4. Why learning rate = 0.001?

- **Initial attempt** (0.01): Caused NaN values after epoch 2
- **Final choice** (0.001): Stable convergence
- **Future**: Adaptive learning rate (Adam optimizer) would improve speed

### 5. Why Cross-Entropy loss?

- **Multi-class classification**: Natural fit for 4 output classes
- **Probabilistic interpretation**: Softmax + Cross-Entropy = clean gradients
- **Alternative**: MSE works but slower convergence

---

## Sources

- Glorot, X., & Bengio, Y. (2010). _Understanding the difficulty of training deep feedforward neural networks_
- Nielsen, M. (2015). _Neural Networks and Deep Learning_
- FEN Notation: [Wikipedia](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation)
