def parse_cli_arguments(args):
    """
    Parse: config_file_1 nb_1 config_file_2 nb_2 ...
    Renvoie une liste: [(config1, nb1), (config2, nb2), ...]
    """
    if len(args) >= 1 and (args[0] == "--help" or args[0] == "-h"):
        print("""USAGE
    ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]

DESCRIPTION
    config_file_i    Configuration file describing the neural network.
    nb_i             Number of networks to generate from this config.""")
        import sys
        sys.exit(0)
    
    if len(args) == 0 or len(args) % 2 != 0:
        raise ValueError("Invalid number of arguments")

    result = []
    for i in range(0, len(args), 2):
        config_file = args[i]
        try:
            nb = int(args[i + 1])
        except ValueError:
            raise ValueError(f"Invalid number: {args[i + 1]}")
        if nb <= 0:
            raise ValueError(f"Number of network must be > 0")
        result.append((config_file, nb))
    return result


def parse_config_file(path):
    """
    Simple key=value parser that ignores comments (#) and normalizes types:
    - input_size -> int
    - layer_sizes -> list[int]
    - activations -> list[str]
    - learning_rate -> float
    """
    config = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                continue
            key, val = map(str.strip, line.split("=", 1))
            config[key] = val

    # Normalize types
    if "input_size" in config:
        try:
            config["input_size"] = int(config["input_size"])
        except ValueError:
            raise ValueError("input_size must be an integer")

    if "layer_sizes" in config:
        raw = config["layer_sizes"]
        # Accept "1,2,3" or [1,2,3] or ["1","2","3"]
        if isinstance(raw, str):
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            try:
                config["layer_sizes"] = [int(p) for p in parts]
            except ValueError:
                raise ValueError("layer_sizes must be comma-separated integers")
        elif isinstance(raw, (list, tuple)):
            try:
                config["layer_sizes"] = [int(p) for p in raw]
            except Exception:
                raise ValueError("layer_sizes list must contain integers")
        else:
            raise ValueError("layer_sizes must be a string or a list of integers")

    if "activations" in config:
        raw = config["activations"]
        # Accept "relu,softmax" or ["relu","softmax"]
        if isinstance(raw, str):
            config["activations"] = [p.strip() for p in raw.split(",") if p.strip()]
        elif isinstance(raw, (list, tuple)):
            config["activations"] = [str(p).strip() for p in raw]
        else:
            raise ValueError("activations must be a string or a list of strings")

    if "learning_rate" in config:
        try:
            config["learning_rate"] = float(config["learning_rate"])
        except ValueError:
            raise ValueError("learning_rate must be a number")

    return config
