def fen_to_vector(fen):
    """Convert FEN to 769-dimensional input vector:
    - 64 squares × 12 piece types (one-hot) = 768
    - 1 for turn (white=1, black=0)
    """
    parts = fen.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Invalid FEN: {fen}")
    
    board_part = parts[0]
    turn = parts[1]
    
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    vector = [0.0] * 769
    
    square = 0
    for char in board_part:
        if char == '/':
            continue
        elif char.isdigit():
            square += int(char)
        else:
            if char in piece_map:
                piece_idx = piece_map[char]
                vector[square * 12 + piece_idx] = 1.0
            square += 1
    
    vector[768] = 1.0 if turn == 'w' else 0.0
    
    return vector

def label_to_vector(label):
    """Convert label to one-hot vector: Nothing, Check, Checkmate, Stalemate
    Handles labels with optional color suffix (e.g., 'Checkmate White')"""
    label = label.strip().lower()
    
    # Extract base label (ignore color suffix)
    base_label = label.split()[0] if ' ' in label else label
    
    labels = {"nothing": 0, "check": 1, "checkmate": 2, "stalemate": 3}
    
    if base_label not in labels:
        raise ValueError(f"Invalid label: {label}")
    
    vector = [0.0] * 4
    vector[labels[base_label]] = 1.0
    return vector

def vector_to_label(vector):
    """Convert output vector to label"""
    labels = ["Nothing", "Check", "Checkmate", "Stalemate"]
    return labels[vector.index(max(vector))]
