import sys
import analyzer.parsor as parsor
import analyzer.train as train
import analyzer.predict as predict

if __name__ == "__main__":
    try:
        arguments = parsor.parse_cli_arguments(sys.argv[1:])
    except Exception as e:
        print(f"error: {e}")
        sys.exit(84)
    
    try:
        parsed_nn = parsor.parse_neural_network_file(sys.argv[1:])
    except Exception as e:
        print(f"error: {e}")
        sys.exit(84)
    
    mode = arguments["mode"]

    if mode == "train":
        try:
            train.train_model(arguments, parsed_nn)
        except Exception as e:
            print(f"error: {e}")
            sys.exit(84)
    elif mode == "predict":
        try:
            predict.predict_model(arguments, parsed_nn)
        except Exception as e:
            print(f"error: {e}")
            sys.exit(84)
    else:
        print("error: Invalid mode specified. Use --train or --predict.")
        sys.exit(84)
