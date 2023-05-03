from torch.utils.data import Dataset
from datasets import load_dataset

class MyDataset(Dataset):
    type = 0
    def __init__(self, split):
        self.dataset = load_dataset(path="csv", data_files=f"./data/{split}.csv", split="train")
        if split =='train':
            type = 0
        else:
            type = 1
    def __len__(self):
        return len(self.dataset)
    def maxlen(self):
        maxlen = 0
        for i in self.dataset:
            if len(i['text']) > maxlen:
                maxlen = len(i['text'])
        return maxlen
    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        label = self.dataset[item]["class"]
        return text, label




if __name__ == '__main__':
    dataset = MyDataset("train")
    print(dataset.type)
    # 最长196
    for i,(text,label) in enumerate(dataset):
        print(label)







