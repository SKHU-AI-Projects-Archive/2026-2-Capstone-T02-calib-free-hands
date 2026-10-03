import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare GigaHands split files.")
    parser.add_argument("--root", default="datasets/gigahands", help="Dataset root.")
    parser.add_argument("--output-dir", default="data/splits/gigahands", help="Directory for split files.")
    return parser.parse_args()


def main():
    parse_args()
    raise NotImplementedError(
        "Split creation policy is not defined yet. Decide the research protocol before generating train/val splits."
    )


if __name__ == "__main__":
    main()
