#! -*- coding:utf-8 -*-
import csv
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch4keras.callbacks import Callback
from bert4torch.snippets import sequence_padding, ListDataset, seed_everything
from bert4torch.optimizers import get_linear_schedule_with_warmup
from bert4torch.layers import LayerNorm
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import defaultdict, deque
from sklearn.metrics import precision_recall_fscore_support

from model import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# 模型参数：训练
epochs = 20  # 训练轮数
steps_per_epoch = None  # 每轮步数
maxlen = 256  # 最大长度
batch_size = 1  # 根据gpu显存设置
learning_rate = 1e-3
bert_learning_rate = 5e-6
warm_factor = 0.1
weight_decay = 0
use_bert_last_4_layers = True
categories = {'BANK': 2, 'PRODUCT': 3, 'COMMENTS_N': 4,'COMMENTS_ADJ': 5 }
label_num = len(categories) + 2

# 模型参数：网络结构
dist_emb_size = 20
type_emb_size = 20
lstm_hid_size = 512
conv_hid_size = 96
bert_hid_size = 768
biaffine_size = 512
ffnn_hid_size = 288
dilation = [1, 2, 3]
emb_dropout = 0.5
conv_dropout = 0.5
out_dropout = 0.33

# BERT base
config_path = '../pre_model/config.json'
checkpoint_path = '../pre_model/pytorch_model.bin'
dict_path = '../pre_model/vocab.txt'


device = torch.device("cuda")


# 用到的小函数
def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def convert_text_to_index(text):
    index, type = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type)


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
dict_path = '../pre_model/vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)
# 相对距离设置
dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9

maxlen = 256  # 最大长度



# 用到的小函数
def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def convert_text_to_index(text):
    index, type = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type)
categories = {'BANK': 2, 'PRODUCT': 3, 'COMMENTS_N': 4,'COMMENTS_ADJ': 5 }
decode_cat = {2:'BANK',   3:'PRODUCT',  4:'COMMENTS_N', 5:'COMMENTS_ADJ'}

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        D = []
        with open(filename) as f:
            reader = csv.reader(f)
            header_row = next(reader)
            # 每一个数据
            for row in reader:
                sentence = [one for one in row[1]]

                tokens = [tokenizer.tokenize(word)[1:-1] for word in sentence[:maxlen - 2]]
                pieces = [piece for pieces in tokens for piece in pieces]
                tokens_ids = [tokenizer._token_start_id] + tokenizer.tokens_to_ids(pieces) + [tokenizer._token_end_id]
                assert len(tokens_ids) <= maxlen
                length = len(tokens)

                # piece和word的对应关系，中文两者一致，除了[CLS]和[SEP]
                _pieces2word = np.zeros((length, len(tokens_ids)), dtype=np.bool)
                e_start = 0
                for i, pieces in enumerate(tokens):
                    if len(pieces) == 0:
                        continue
                    pieces = list(range(e_start, e_start + len(pieces)))
                    _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                    e_start += len(pieces)

                # 相对距离
                _dist_inputs = np.zeros((length, length), dtype=np.int)
                for k in range(length):
                    _dist_inputs[k, :] += k
                    _dist_inputs[:, k] -= k

                for i in range(length):
                    for j in range(length):
                        if _dist_inputs[i, j] < 0:
                            _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                        else:
                            _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
                _dist_inputs[_dist_inputs == 0] = 19

                _grid_mask2d = np.ones((length, length), dtype=np.bool)
                D.append((tokens_ids, _pieces2word, _dist_inputs, _grid_mask2d))
        return D


def collate_fn(data):
    tokens_ids, pieces2word, dist_inputs, grid_mask2d = map(list, zip(*data))

    sent_length = torch.tensor([i.shape[0] for i in pieces2word], dtype=torch.long, device=device)
    # max_wordlen: word长度，非token长度，max_tokenlen：token长度
    max_wordlen = torch.max(sent_length).item()
    max_tokenlen = np.max([len(x) for x in tokens_ids])
    tokens_ids = torch.tensor(sequence_padding(tokens_ids), dtype=torch.long, device=device)
    batch_size = tokens_ids.size(0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = torch.tensor(x, dtype=torch.long, device=device)
        return new_data

    dis_mat = torch.zeros((batch_size, max_wordlen, max_wordlen), dtype=torch.long, device=device)
    dist_inputs = fill(dist_inputs, dis_mat)
    mask2d_mat = torch.zeros((batch_size, max_wordlen, max_wordlen), dtype=torch.bool, device=device)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_wordlen, max_tokenlen), dtype=torch.bool, device=device)
    pieces2word = fill(pieces2word, sub_mat)

    # return [tokens_ids, pieces2word, dist_inputs, sent_length, grid_mask2d, grid_labels, _entity_text]
    return [tokens_ids, pieces2word, dist_inputs, sent_length, grid_mask2d]





model = Model(use_bert_last_4_layers).to(device)






test_dataloader = DataLoader(MyDataset('../data/baidu/test.csv'),
                              batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
checkpoint = torch.load("./checkpoint/new_best__model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print(start_epoch)
res = []


model.eval()
for data_batch in tqdm(test_dataloader, desc='Evaluate'):
    (token_ids, pieces2word, dist_inputs, sent_length, grid_mask2d) = data_batch
    outputs = model.predict([token_ids, pieces2word, dist_inputs, sent_length, grid_mask2d])
    grid_mask2d = grid_mask2d.clone()

    outputs = torch.argmax(outputs, -1)
    _grid_labels = outputs[0].tolist()
    pred_label = []
    for i in range(len(_grid_labels)):
        for j in range(len(_grid_labels)):
            if _grid_labels[i][j] > 1:
                pred_start = j
                pred_end = i
                pred_type = decode_cat[_grid_labels[i][j]]
                pred_label.append([pred_start,pred_end,pred_type])
    pred_flag = []
    length = len(outputs[0])
    for i in range(length):
        pred_flag.extend('O')
    for entity in pred_label:
        pred_flag[entity[0]] = 'B-'+entity[2]
        for j in range(entity[0]+1,entity[1]+1):
            pred_flag[j] = 'I-' + entity[2]
    str = pred_flag[0]
    for i in range(1,length):
        str += ' '
        str += pred_flag[i]

    res.append(str)


with open("./res/new_best_model.csv", "w", newline='', encoding='utf-8') as file:
    w = csv.writer(file)
    for row in res:
        w.writerow([row])
