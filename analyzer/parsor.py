import sys
import json

def parse_cli_arguments(args):
    if len(args) >= 1 and (args[0] == "--help" or args[0] == "-h"):
        print("""USAGE
    ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE FILE

DESCRIPTION
    --train     Launch in training mode. FILE contains FEN positions and labels.
    --predict   Launch in prediction mode. FILE contains FEN positions.
    --save      Save network to SAVEFILE (train mode only).
    LOADFILE    File containing the neural network.
    FILE        File containing chessboards in FEN notation.""")
        sys.exit(0)
    
    mode = None
    save_file = None
    load_file = None
    data_file = None
    
    i = 0
    while i < len(args):
        if args[i] == "--train":
            mode = "train"
        elif args[i] == "--predict":
            mode = "predict"
        elif args[i] == "--save":
            if i + 1 >= len(args):
                raise ValueError("--save requires a filename")
            save_file = args[i + 1]
            i += 1
        elif load_file is None:
            load_file = args[i]
        elif data_file is None:
            data_file = args[i]
        i += 1
    
    if not mode or not load_file or not data_file:
        raise ValueError("Missing required arguments")
    
    return {
        "mode": mode,
        "load_file": load_file,
        "data_file": data_file,
        "save_file": save_file if save_file else load_file
    }

def parse_neural_network_file(args):
    arguments = parse_cli_arguments(args)
    with open(arguments["load_file"], "r") as f:
        network = json.load(f)
    return network