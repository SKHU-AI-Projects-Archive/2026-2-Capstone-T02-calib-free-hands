import argparse

from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Future AnyCalib training entry point.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    experiment_name = config.get("experiment", {}).get("name", "<unnamed>")
    print(f"Loaded config for training experiment: {experiment_name}")
    raise NotImplementedError(
        "AnyCalib training is not wired yet. Connect the official model and dataset protocol first."
    )


if __name__ == "__main__":
    main()
