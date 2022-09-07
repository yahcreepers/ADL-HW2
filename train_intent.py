#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2021/4/14 10:30 上午
# @Author  : hanxiuwei
"""
import csv
import json
import random
import logging
import numpy as np

from argparse import ArgumentParser, Namespace
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, AdamW, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

logger = logging.getLogger('classification')
E = 0

class BertClassificationModel:
    """创建分类模型类，使用transforms库中的BertForSequenceClassification类，并且该类自带loss计算"""
    def __init__(self, train, validation, vocab_path, config_path, pretrain_model_path, save_model_path,
                 learning_rate, n_class, epochs, batch_size, val_batch_size, max_len, gpu=True):
        super(BertClassificationModel, self).__init__()
        # 类别数
        self.n_class = n_class
        # 句子最大长度
        self.max_len = max_len
        # 学习率
        self.lr = learning_rate
        # 训练轮数
        self.epochs = epochs
        # 训练集的batch_size
        self.batch_size = batch_size
        # 验证集的batch_size
        self.val_batch_size = val_batch_size
        # 模型存储位置
        self.save_model_path = save_model_path
        # 是否使用gpu
        self.gpu = gpu

        # 加载bert分词模型词典
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)
        # 加载bert模型配置信息
        config = BertConfig.from_json_file(config_path)
        # 设置分类模型的输出个数
        config.num_labels = n_class
        # 加载训练数据集
        self.train = self.load_data(train)
        # 加载测试数据集
        self.validation = self.load_data(validation)
        # 加载bert分类模型
        self.model = BertForSequenceClassification.from_pretrained(pretrain_model_path, config=config)
        # 设置GPU
        if self.gpu:
            seed = 42
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            self.device = torch.device(f'cuda:{args.cuda}')
        else:
            self.device = 'cpu'

    def encode_fn(self, text_lists):
        """
        训练：将text_list embedding成bert模型可用的输入形式
        :param text_lists:['我爱你','猫不是狗']
        :return:
        """
        # 返回的类型为pytorch tensor
        tokenizer = self.tokenizer(
            text_lists,
            padding=True,
            truncation=True,
            #max_length=self.max_len,
            #padding='max_length',
            return_tensors='pt'
        )
        input_ids = tokenizer['input_ids']
        token_type_ids = tokenizer['token_type_ids']
        attention_mask = tokenizer['attention_mask']
        return input_ids, token_type_ids, attention_mask

    def load_data(self, path):
        """
        训练：处理训练的csv文件
        :param path:
        :return:
        """
        text_lists = []
        labels = []
        file = open(path, encoding='utf-8')
        f = open("./cache/intent/intent2idx.json", "r")
        intent2idx = json.load(f)
        dataset = json.load(file)
        for data in dataset:
            # 这里可以改，label在什么位置就改成对应的index
            label = intent2idx[data["intent"]]
            text = data["text"]
            text_lists.append(text)
            labels.append(label)
        #print(text_lists)
        input_ids, token_type_ids, attention_mask = self.encode_fn(text_lists)
        labels = torch.tensor(labels)
        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels)
        return data

    @staticmethod
    def flat_accuracy(predicts, labels):
        """
        训练：计算准确率得分
        :param predicts:
        :param labels:
        :return:
        """
        pred_flat = np.argmax(predicts, axis=1).flatten()
        labels_flat = labels.flatten()
        return accuracy_score(labels_flat, pred_flat)

    def train_model(self):
        """
        训练：训练模型
        :return:
        """
        if self.gpu:
            self.model.cuda(args.cuda)
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        # 处理成多个batch的形式
        train_data = DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True)
        val_data = DataLoader(
            self.validation,
            batch_size=self.val_batch_size,
            shuffle=True)

        total_steps = len(train_data) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        max = float("-inf")
        for epoch in range(self.epochs):
            self.model.train()
            total_loss, total_val_loss = 0, 0
            total_eval_accuracy = 0
            print('epoch:', epoch, ', step_number:', len(train_data))
            # 训练，其中step是迭代次数
            # 每一次迭代都是一次权重更新，每一次权重更新需要batch_size个数据进行Forward运算得到损失函数，再BP算法更新参数。
            # 1个iteration等于使用batch_size个样本训练一次
            for step, batch in enumerate(train_data):
                self.model.zero_grad()
                # 输出loss 和 每个分类对应的输出，softmax后才是预测是对应分类的概率
                outputs = self.model(input_ids=batch[0].to(self.device),
                                     token_type_ids=batch[1].to(self.device),
                                     attention_mask=batch[2].to(self.device),
                                     labels=batch[3].to(self.device))
                loss, logits = outputs[0], outputs[1]
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                # 每100步输出一下训练的结果，flat_accuracy()会对logits进行softmax
                if step % 100 == 0 and step > 0:
                    self.model.eval()
                    logits = logits.detach().cpu().numpy()
                    label_ids = batch[3].cuda(args.cuda).data.cpu().numpy()
                    avg_val_accuracy = self.flat_accuracy(logits, label_ids)
                    print(f'step:{step}')
                    print(f'Accuracy: {avg_val_accuracy:.4f}')
                    print('*'*20)
            # 每个epoch结束，就使用validation数据集评估一次模型
            self.model.eval()
            print('testing ....')
            for i, batch in enumerate(val_data):
                with torch.no_grad():
                    outputs = self.model(input_ids=batch[0].to(self.device),
                                         token_type_ids=batch[1].to(self.device),
                                         attention_mask=batch[2].to(self.device),
                                         labels=batch[3].to(self.device))
                    loss, logits = outputs[0], outputs[1]
                    total_val_loss += loss.item()

                    logits = logits.detach().cpu().numpy()
                    label_ids = batch[3].cuda(args.cuda).data.cpu().numpy()
                    total_eval_accuracy += self.flat_accuracy(
                        logits, label_ids)

            avg_train_loss = total_loss / len(train_data)
            avg_val_loss = total_val_loss / len(val_data)
            avg_val_accuracy = total_eval_accuracy / len(val_data)

            print(f'Train loss     : {avg_train_loss}')
            print(f'Validation loss: {avg_val_loss}')
            print(f'Accuracy: {avg_val_accuracy:.4f}')
            print('*'*20)
            if(avg_val_accuracy > max):
                max = avg_val_accuracy
                E = epoch
                self.save_model(self.save_model_path)

    def save_model(self, path):
        """
        训练：保存分词模型和分类模型
        :param path:
        :return:
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @staticmethod
    def load_model(path):
        """
        预测：加载分词模型和分类模型
        :param path:
        :return:
        """
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = BertForSequenceClassification.from_pretrained(path)
        return tokenizer, model

    @staticmethod
    def load_data_predict(path):
        """
        预测：加载测试数据
        :param path:
        :return:
        """
        text_lists = []
        labels = []
        file = open(path, encoding='utf-8')
        f = open("./cache/intent/intent2idx.json", "r")
        intent2idx = json.load(f)
        dataset = json.load(file)
        for data in dataset:
            # 这里可以改，label在什么位置就改成对应的index
            label = intent2idx[data["intent"]]
            text = data["text"]
            text_lists.append(text)
            labels.append(label)
        return text_lists, labels
    @staticmethod
    def load_data_test(path):
        """
        预测：加载测试数据
        :param path:
        :return:
        """
        text_lists = []
        id = []
        file = open(path, encoding='utf-8')
        dataset = json.load(file)
        for data in dataset:
            # 这里可以改，label在什么位置就改成对应的index
            text = data["text"]
            text_lists.append(text)
            id.append(data["id"])
        return id, text_lists
    def predict_model(self, tokenizer, model, id, text_lists):
        """
        预测：输出模型的召回率、准确率、f1-score
        :param tokenizer:
        :param model:
        :param text_lists:
        :param y_real:
        :return:
        """
        preds = self.predict_batch(tokenizer, model, text_lists)
        #print(preds)
        file = open("pred.intent.csv", "w")
        writer = csv.writer(file)
        f = open("./cache/intent/intent2idx.json", "r")
        intent2idx = json.load(f)
        idx2intent = {intent2idx[i]:i for i in intent2idx}
        writer.writerow(["id", "intent"])
        for i in range(len(preds)):
            writer.writerow([id[i], idx2intent[preds[i]]])

    def eval_model(self, tokenizer, model, text_lists, y_real):
        """
        预测：输出模型的召回率、准确率、f1-score
        :param tokenizer:
        :param model:
        :param text_lists:
        :param y_real:
        :return:
        """
        preds = self.predict_batch(tokenizer, model, text_lists)
        #print(preds)
        print(classification_report(y_real, preds))
    def predict_batch(self, tokenizer, model, text_lists):
        """
        预测：预测
        :param tokenizer:
        :param model:
        :param text_lists:
        :return:
        """
        tokenizer = tokenizer(
            text_lists,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'  # 返回的类型为pytorch tensor
        )
        input_ids = tokenizer['input_ids']
        token_type_ids = tokenizer['token_type_ids']
        attention_mask = tokenizer['attention_mask']
        pred_data = TensorDataset(input_ids, token_type_ids, attention_mask)
        pred_dataloader = DataLoader(
            pred_data, batch_size=self.batch_size, shuffle=False)
        model = model.to(self.device)
        model.eval()
        preds = []
        for i, batch in enumerate(pred_dataloader):
            with torch.no_grad():
                outputs = model(input_ids=batch[0].to(self.device),
                                token_type_ids=batch[1].to(self.device),
                                attention_mask=batch[2].to(self.device)
                                )
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                preds += list(np.argmax(logits, axis=1))
        return preds


def load_data_test(path):
    """
    预测：加载测试数据
    :param path:
    :return:
    """
    text_lists = []
    id = []
    file = open(path, encoding='utf-8')
    dataset = json.load(file)
    for data in dataset:
        # 这里可以改，label在什么位置就改成对应的index
        text = data["text"]
        text_lists.append(text)
        id.append(data["id"])
    return id, text_lists

def main(args):
    EPOCH = 20
    # 预训练模型的存储位置为 ../../pretrained_models/bert-base-chinese/
    # 分类模型和分词模型的存储位置是 ../trained_model/bert_model/
    bert_model = BertClassificationModel(
        train='./data/intent/train.json',
        validation='./data/intent/eval.json',
        vocab_path='./trained_model/bert_model/vocab.txt',
        config_path='./trained_model/bert_model/config.json',
        pretrain_model_path='./trained_model/bert_model/pytorch_model.bin',
        save_model_path='./trained_model/bert_model-1',
        learning_rate=2e-5,
        n_class=150,
        epochs=args.num_epoch,
        batch_size=4,
        val_batch_size=4,
        max_len=50,
        gpu=True)
    # 模型训练
    if args.do_train:
        bert_model.train_model()

    # 模型预测
    if args.do_eval:
        classification_tokenizer, classification_model = bert_model.load_model(
            bert_model.save_model_path)
        text_list, y_true = bert_model.load_data_predict('./data/intent/eval.json')
        bert_model.eval_model(classification_tokenizer, classification_model, text_list, y_true)
    if args.do_predict:
        if not args.do_eval:
            classification_tokenizer, classification_model = bert_model.load_model(args.load_model_path)
        id, text_list = load_data_test('./data/intent/test.json')
        #print(text_list)
        bert_model.predict_model(classification_tokenizer, classification_model, id, text_list)

def parse_args() -> Namespace:
    parser = ArgumentParser()

    # data
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--num_epoch", type=int, default=3)
    parser.add_argument("--load_model_path", type=str)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)

