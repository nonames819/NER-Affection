import os
import numpy as np
import pandas as pd
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from torch.utils.data import DataLoader
from model import BertClassifier
from transformers import BertTokenizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import csv
tokenizer = BertTokenizer.from_pretrained("../pre_model")




from torch.utils.data import Dataset
from datasets import load_dataset

class MyDataset(Dataset):
    def __init__(self, split):
        self.dataset = load_dataset(path="csv", data_files=f"../data/baidu/{split}.csv", split="train")
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        return text



def collate_fn(data):
    sentes = [i for i in data]
    # 编码
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sentes,
                                       truncation=True,
                                       padding="max_length",
                                       max_length=196,
                                       return_tensors="pt",
                                       return_length=True)

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]

    return input_ids, attention_mask, token_type_ids

# 创建数据集
test_dataset = MyDataset("test")

# 创建dataloader
test_loader = DataLoader(dataset=test_dataset,
                        batch_size=16,
                        shuffle=False,
                        drop_last=False,
                        collate_fn=collate_fn)


if __name__ == '__main__':
    print(DEVICE)


    model = BertClassifier().cuda()
    softmax = nn.Softmax(dim=1)

    checkpoint = torch.load("./checkpoint/5e-6_10_lastmodel.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    # 模型测试

    model.eval()
    pred = []
    print('--------------------- TEST ---------------------')
    for i, (input_ids, attention_mask, token_type_ids) in enumerate(test_loader):
        input_ids, attention_mask, token_type_ids = input_ids.cuda(), \
                                                    attention_mask.cuda(), \
                                                    token_type_ids.cuda()

        with torch.no_grad():
             logits = model(input_ids, attention_mask)
        out_max = logits.argmax(dim=1)
        pred.extend(out_max.tolist())

    # with open('./res/classify.csv', "w", newline='') as f:
    #     writer = csv.writer(f)
    #     for row in pred:
    #         writer.writerow(row)
    # 将list转为dataframe 显然就变成一列了
    d = pd.DataFrame(pred)
    d.to_csv('./res/5e-6_10_classify.csv', mode='w', header=None)  # mode表示追加 在追加时会将列名也作为一行进行追加，故header隐藏表头（列名）
