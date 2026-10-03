import argparse

from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Future AnyCalib evaluation entry point.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    experiment_name = config.get("experiment", {}).get("name", "<unnamed>")
    print(f"Loaded config for evaluation experiment: {experiment_name}")
    raise NotImplementedError(
        "AnyCalib evaluation is not wired yet. Connect the official model and metric definition first."
    )


if __name__ == "__main__":
    main()
