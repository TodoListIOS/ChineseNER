# coding=utf-8
import pickle
import pdb


def save_to_file(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()


with open('../data/Bosondata.pkl', 'rb') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)
    x_train = pickle.load(inp)
    y_train = pickle.load(inp)
    x_test = pickle.load(inp)
    y_test = pickle.load(inp)
    x_valid = pickle.load(inp)
    y_valid = pickle.load(inp)
# save_to_file('./word2id_2.txt', str(word2id))
# print(word2id)
# print(id2word)
# print(tag2id)
# print(id2tag)
# print('word2id len:', len(word2id))
# print('id2word len:', len(id2word))
# print('tag2id len:', len(tag2id))
# print('id2tag len:', len(id2tag))
# print("x_train len:", len(x_train))
# print('y_train len:', len(y_train))
# print("x_test len:", len(x_test))
# print('y_test len:', len(y_test))
# print("x_valid len", len(x_valid))
# print('y_valid len:', len(y_valid))
"""原始数据完整的2000行
word2id len: 3435
id2word len: 3434
tag2id len: 20
id2tag len: 20
x_train len: 10721
y_train len: 10721
x_test len: 3351
y_test len: 3351
x_valid len 2681
y_valid len: 2681
"""

"""原始数据减少为100行后
word2id len: 1447
id2word len: 1446
tag2id len: 20
id2tag len: 20
x_train len: 435
y_train len: 435
x_test len: 137
y_test len: 137
x_valid len 109
y_valid len: 109
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
from pytorch.BiLSTM_CRF import BiLSTM_CRF
from pytorch.resultCal import calculate
import numpy as np

#############
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100
HIDDEN_DIM = 200
EPOCHS = 30  # 和tensorflow版本一样设置成30

tag2id[START_TAG] = len(tag2id)
tag2id[STOP_TAG] = len(tag2id)

model = BiLSTM_CRF(len(word2id) + 1, tag2id, EMBEDDING_DIM, HIDDEN_DIM)

optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=1e-4)

for epoch in range(EPOCHS):
    index = 0
    for sentence, tags in zip(x_train, y_train):
        index += 1
        model.zero_grad()

        # # 自己加的，便于调试
        # # 此时sentence还是list
        # origin_sentence = []
        # for s in sentence:
        #     if s == 0:  # 略过补0的部分
        #         continue
        #     ch = id2word[s]
        #     origin_sentence.append(ch)
        #
        # origin_tags = []
        # for t in tags:
        #     if t == 0:  # 略过补0的部分
        #         continue
        #     ch = id2tag[t]
        #     origin_tags.append(ch)

        sentence = torch.tensor(sentence, dtype=torch.long)  # 把list转成tensor
        tags = torch.tensor([tag2id[t] for t in tags], dtype=torch.long)  # 把list转成tensor

        # sentence中非0的个数 == tags中非0的个数

        # 前向传播
        loss = model.neg_log_likelihood(sentence, tags)

        # 计算loss,梯度， 通过调用optimizer.step()来更新模型的参数
        loss.backward()
        optimizer.step()

        if index % 300 == 0:
            print("epoch", epoch, "index", index)
    # 一个epoch训练结束

    entityres = []
    entityall = []
    for sentence, tags in zip(x_test, y_test):
        # print('tags: {}'.format(str(tags)))  # tags是list类型
        # # 自己加的，便于调试
        # # 此时sentence还是list
        origin_sentence = []
        for s in sentence:
            if s == 0:  # 略过补0的部分
                continue
            ch = id2word[s]
            origin_sentence.append(ch)
        # print('origin_sentence: {}'.format(str(origin_sentence)))

        sentence = torch.tensor(sentence, dtype=torch.long)  # 转换后sentence是tensor类型
        # print('Transfer sentence list to sentence tensor, sentence: {}'.format(str(sentence)))
        score, predict = model(sentence)
        # print('predict: {}'.format(str(predict)))  # predict 是list类型

        # 把tensor型的sentence转成list型
        sentence = sentence.numpy().tolist()
        # print('transfer tensor sentence to list sentence, sentence: {}'.format(str(sentence)))

        origin_tags = []
        for t in tags:
            if t == 0:  # 略过补0的部分
                continue
            ch = id2tag[t]
            origin_tags.append(ch)

        debug = 1
        entityres = calculate(origin_sentence, origin_tags, sentence, predict, id2word, id2tag, entityres)
        # print('epoch{} entity_res: {}'.format(str(epoch), str(entityres)))
        debug = 2
        entityall = calculate(origin_sentence, origin_tags, sentence, tags, id2word, id2tag, entityall)
        debug = 3
        # print('epoch{} entity_all: {}'.format(str(epoch), str(entityall)))

    # print('epoch{} Finish all test data.'.format(str(epoch)))
    debug = 4
    jiaoji = [i for i in entityres if i in entityall]
    if len(jiaoji) != 0:
        accuracy = float(len(jiaoji)) / len(entityres)
        recall = float(len(jiaoji)) / len(entityall)
        print("test:")
        print("Accuracy:", accuracy)
        print("Recall:", recall)
        print("F1:", (2 * accuracy * recall) / (accuracy + recall))
    else:
        print("Accuracy:", 0)

    path_name = "./model/model" + str(epoch) + ".pkl"
    # print(path_name)
    torch.save(model, path_name)
    print("epoch{} model has been saved to {}\n".format(str(epoch), path_name))
