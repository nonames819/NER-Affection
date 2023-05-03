import os
import numpy as np
import torch.nn as nn
from scipy.stats import pearsonr
from torch import optim

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from MyData import MyDataset
from torch.utils.data import DataLoader
from net import Model
from transformers import BertTokenizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 30


tokenizer = BertTokenizer.from_pretrained("../pre_model")

