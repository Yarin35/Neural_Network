import json
from analyzer.fen_parser import fen_to_vector, label_to_vector
from analyzer.network import forward_pass, backward_pass, cross_entropy_loss

def train_model(arguments, network):
    data_file = arguments["data_file"]
    save_file = arguments["save_file"]
    learning_rate = network["meta"].get("learning_rate", 0.01)
    
    # Load training data
    training_data = []
    with open(data_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            # Format: FEN (6 parts) + label [+ optional color]
            # Example: 8/8/8/... w - - 4 70 Checkmate [Black]
            fen = " ".join(parts[:6])
            label = " ".join(parts[6:])  # Keep entire label (with optional color)
            try:
                input_vec = fen_to_vector(fen)
                target_vec = label_to_vector(label)
                training_data.append((input_vec, target_vec))
            except Exception:
                continue
    
    if not training_data:
        raise ValueError("No valid training data found")
    
    print(f"Training on {len(training_data)} samples...")
    
    # Training loop - adjust epochs based on dataset size
    if len(training_data) > 1000:
        epochs = 20
    else:
        epochs = 100
        
    for epoch in range(epochs):
        total_loss = 0.0
        
        for input_vec, target_vec in training_data:
            output, cache = forward_pass(network, input_vec)
            loss = cross_entropy_loss(output, target_vec)
            total_loss += loss
            
            backward_pass(network, cache, target_vec, learning_rate)
        
        avg_loss = total_loss / len(training_data)
        if epoch % 2 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save trained network
    with open(save_file, "w") as f:
        json.dump(network, f, indent=4)
    
    print(f"Training complete. Network saved to {save_file}")
