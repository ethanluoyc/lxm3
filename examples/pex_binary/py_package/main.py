import argparse

parser = argparse.ArgumentParser(description="A simple example.")
parser.add_argument("--seed", type=int, help="Random seed.")
parser.add_argument("--data", type=str)


def main():
    args = parser.parse_args()
    print(args)
    with open(args.data, "r") as f:
        print("data file contents:")
        print(f.read())


if __name__ == "__main__":
    main()
