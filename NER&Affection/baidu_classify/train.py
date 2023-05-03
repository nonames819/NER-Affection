import os
import numpy as np
import torch.nn as nn
from scipy.stats import pearsonr
from torch import optim

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from MyData import MyDataset
from torch.utils.data import DataLoader
from model import BertClassifier
from transformers import BertTokenizer
DEVICE = torch.device("cuda")
EPOCH = 10



tokenizer = BertTokenizer.from_pretrained("../pre_model")

def collate_fn(data):

    sentes = [i[0] for i in data]
    label = [i[1] for i in data]
    # 编码
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sentes,
                                       truncation=True,
                                       padding="max_length",
                                       max_length=161,
                                       return_tensors="pt",
                                       return_length=True)

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    labels = torch.tensor(label)

    return input_ids, attention_mask, token_type_ids, labels

# 创建数据集
train_dataset = MyDataset("train")

# 创建dataloader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=16,
                          shuffle=True,
                          drop_last=True,
                          collate_fn=collate_fn)



if __name__ == '__main__':
    print(DEVICE)

    best_acc = 0
    model = BertClassifier().cuda()
    softmax = nn.Softmax(dim=1)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 0.000005, betas=(0.99, 0.999), weight_decay=5e-5)

    checkpoint = torch.load("./checkpoint/5e-6_8_lastmodel.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

    model.train()
    for epoch in range(start_epoch,EPOCH):
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, labels = input_ids.cuda(), \
                                                            attention_mask.cuda(), \
                                                            token_type_ids.cuda(), \
                                                            labels.cuda()

            logits = model(input_ids, attention_mask)
            loss = loss_func(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 ==0:

                out_max = logits.argmax(dim=1)
                acc = (out_max == labels).sum() / len(labels)
                print(f'train ==> epoch{epoch} batch{i}, acc{acc}, loss{loss}')

        if epoch == EPOCH - 1 :
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f"./checkpoint/5e-6_10_lastmodel.pth")

