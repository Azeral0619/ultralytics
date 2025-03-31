def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Change class names of a trained model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file.")
    parser.add_argument(
        "--names", type=str, nargs="+", required=True, help="New class names list, e.g. --names aaa bbb ccc"
    )
    parser.add_argument("--output", type=str, default=None, help="Path to save the new model.")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model_path = args.model
    names = args.names
    # list to dict
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    import torch

    try:
        model = torch.load(model_path)
    except FileNotFoundError:
        print("Error: model file not found.")
        exit(1)
    ori = model["model"].names
    if ori is None or len(ori) == 0:
        print("Error: model does not have class names.")
        exit(1)
    if len(ori) != len(names):
        print("Error: the number of class names does not match.")
        exit(1)
    model["model"].names = names
    if args.output is None:
        output_path = model_path.replace(".pt", "_zhcn.pt") if model_path.endswith(".pt") else f"{model_path}_zhcn.pt"
        torch.save(model, output_path)
    else:
        torch.save(model, args.output)
    print(f"change {ori} to {names}")
