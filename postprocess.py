import json
import csv
from argparse import ArgumentParser, Namespace

def main(args):
    file = open(args.file, encoding='utf-8')
    data = json.load(file)
    header = ["id", "answer"]
    result = open(args.out, "w")
    writer = csv.writer(result)
    writer.writerow(header)
    for i in data:
        writer.writerow([i, data[i]])

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--file", type=str, default="")
    parser.add_argument("--out", type=str, default="result.csv")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
