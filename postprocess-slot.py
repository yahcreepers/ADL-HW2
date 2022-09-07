import json
import csv
from argparse import ArgumentParser, Namespace

def main(args):
    file = open(args.file, encoding='utf-8')
    header = ["id", "tags"]
    result = open("pred.slot.csv", "w")
    writer = csv.writer(result)
    writer.writerow(header)
    i = 0
    for data in file.readlines():
        writer.writerow([f"test-{i}", data[:-1]])
        i += 1

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--file", type=str, default="")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
