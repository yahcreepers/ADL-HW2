import json
from argparse import ArgumentParser, Namespace
import csv

def main(args):
    if(args.do_train):
        S = ["train", "valid"]
        dataset = {}
        context_file = open(args.context, encoding='utf-8')
        train_file = open(args.train, encoding='utf-8')
        valid_file = open(args.valid, encoding='utf-8')
        context = json.load(context_file)
        
        dataset["train"] = json.load(train_file)
        dataset["valid"] = json.load(valid_file)
        #print(dataset["train"][0])
    #    print(context[0])
    #    print(train[0])
        for s in S:
            file = open(s + "_pro.json", "w")
            for data in dataset[s]:
                D = {}
                D["id"] = data["id"]
                D["sent1"] = data["question"]
                D["question"] = data["question"]
                D["label"] = data["paragraphs"].index(data["relevant"])
                D["context"] = context[data["relevant"]]
                D["answers"] = {"answer_start": [data["answer"]["start"]], "text": [data["answer"]["text"]]}
                for i in range(4):
                    D[f"ending{i}"] = context[data["paragraphs"][i]]
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
        if(not args.do_train):
            context_file = open(args.context, encoding='utf-8')
            context = json.load(context_file)
        test_file = open(args.test, encoding='utf-8')
        dataset = json.load(test_file)
        file = open("test_pro.json", "w")
        for data in dataset:
            D = {}
            D["id"] = data["id"]
            D["sent1"] = data["question"]
            D["question"] = data["question"]
            D["label"] = 0
            D["context"] = ""
            D["answers"] = {"answer_start": [0], "text": [""]}
            for i in range(4):
                D[f"ending{i}"] = context[data["paragraphs"][i]]
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
    parser.add_argument("--train", type=str, default="train.json")
    parser.add_argument("--valid", type=str, default="valid.json")
    parser.add_argument("--test", type=str, default="test.json")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

