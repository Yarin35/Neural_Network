from analyzer.fen_parser import fen_to_vector, vector_to_label, label_to_vector
from analyzer.network import forward_pass

def predict_model(arguments, network):
    data_file = arguments["data_file"]
    
    total = 0
    correct = 0
    
    with open(data_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 7:
                # Format: FEN (6 parts) + label (1-2 words)
                fen = " ".join(parts[:6])
                expected = " ".join(parts[6:])
                has_expected = True
            else:
                fen = line
                has_expected = False
            
            try:
                input_vec = fen_to_vector(fen)
                output, _ = forward_pass(network, input_vec)
                prediction = vector_to_label(output)
                
                # Add color for Check/Checkmate (color indicates who is in check)
                if prediction in ["Check", "Checkmate"]:
                    parts = fen.split()
                    if len(parts) >= 2:
                        # The side to move is the one in check/checkmate
                        color = "White" if parts[1] == 'W' else "Black"
                        prediction = f"{prediction} {color}"
                
                if has_expected:
                    total += 1
                    is_correct = (prediction == expected)
                    if is_correct:
                        correct += 1
                    status = "✓" if is_correct else "✗"
                    print(f"{status} {prediction} (expected: {expected})")
                else:
                    print(prediction)
            except Exception as e:
                print(f"Error processing FEN: {e}")
    
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n{'='*50}")
        print(f"Results: {correct}/{total} correct ({accuracy:.2f}%)")
        print(f"{'='*50}")
