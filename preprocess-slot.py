import json
from argparse import ArgumentParser, Namespace
import csv

def main(args):
    if(args.do_train):
        S = ["train", "valid"]
        dataset = {}
        train_file = open(args.train, encoding='utf-8')
        valid_file = open(args.valid, encoding='utf-8')
        
        dataset["train"] = json.load(train_file)
        dataset["valid"] = json.load(valid_file)
        #print(dataset["train"][0])
    #    print(context[0])
    #    print(train[0])
        for s in S:
            file = open(s + "_slot_pro.json", "w")
            for data in dataset[s]:
                D = {}
                D["id"] = data["id"]
                D["words"] = data["tokens"]
                D["ner"] = data["tags"]
                json.dump(D, file, ensure_ascii=False)
            file.close()
    #    for s in S:
    #        file = open(s + '_pro.csv', "w")
    #        writer = csv.writer(file)
    #        header = dataset[s][0].keys()
    #        writer.writerow(header)
    #        for data in dataset[s]:
    #            writer.writerow(data.values())
                
    #    Train = []
    #    for i, data in enumerate(dataset["train"]):
    #        Train.append(data)
    #    Valid = []
    #    for i, data in enumerate(dataset["valid"]):
    #        Valid.append(data)
#        file = open("train_pro.json", "w")
#        for data in dataset["train"]:
#            json.dump(data, file, ensure_ascii=False)
#        #json.dump(a, file, ensure_ascii=False)
#        file.close()
#        file = open("valid_pro.json", "w")
#        for data in dataset["valid"]:
#            json.dump(data, file, ensure_ascii=False)
#        #json.dump(data, file, ensure_ascii=False)
#        file.close()
    #    print(len(dataset["train"]))
    #    print(len(dataset["valid"]))
    if(args.do_predict):
        test_file = open(args.test, encoding='utf-8')
        dataset = json.load(test_file)
        file = open("test_slot_pro.json", "w")
        for data in dataset:
            D = {}
            D["id"] = data["id"]
            D["words"] = data["tokens"]
            D["ner"] = ["O" for i in range(len(data["tokens"]))]
            json.dump(D, file, ensure_ascii=False)
        file.close()
#        file = open("test_pro.json", "w")
#        for data in dataset:
#            json.dump(data, file, ensure_ascii=False)
#        #json.dump(a, file, ensure_ascii=False)
#        file.close()
        

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--context", type=str, default="context.json")
    parser.add_argument("--train", type=str, default="./data/slot/train.json")
    parser.add_argument("--valid", type=str, default="./data/slot/eval.json")
    parser.add_argument("--test", type=str, default="./data/slot/test.json")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)


