import torch
import torch.nn.functional as F
import torch.nn as nn
from bert4torch.layers import LayerNorm
from bert4torch.models import build_transformer_model, BaseModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence




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




# 定义bert上的模型结构
class ConvolutionLayer(nn.Module):
    '''卷积层
    '''

    def __init__(self, input_size, channels, dilation, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        )

        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)

        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = F.gelu(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs


class Biaffine(nn.Module):
    '''仿射变换
    '''

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s


class MLP(nn.Module):
    '''MLP全连接
    '''

    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class CoPredictor(nn.Module):
    def __init__(self, cls_num, hid_size, biaffine_size, channels, ffnn_hid_size, dropout=0):
        super().__init__()
        self.mlp1 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.mlp2 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout)
        self.linear = nn.Linear(ffnn_hid_size, cls_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, z):
        h = self.dropout(self.mlp1(x))
        t = self.dropout(self.mlp2(y))
        o1 = self.biaffine(h, t)

        z = self.dropout(self.mlp_rel(z))
        o2 = self.linear(z)
        return o1 + o2


class Model(BaseModel):
    def __init__(self, use_bert_last_4_layers=False):
        super().__init__()
        self.use_bert_last_4_layers = use_bert_last_4_layers
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path,
                                            # segment_vocab_size=0,
                                            output_all_encoded_layers=True if use_bert_last_4_layers else False)
        lstm_input_size = self.bert.configs['hidden_size']

        self.dis_embs = nn.Embedding(20, dist_emb_size)
        self.reg_embs = nn.Embedding(3, type_emb_size)

        self.encoder = nn.LSTM(lstm_input_size, lstm_hid_size // 2, num_layers=1, batch_first=True,
                               bidirectional=True)

        conv_input_size = lstm_hid_size + dist_emb_size + type_emb_size

        self.convLayer = ConvolutionLayer(conv_input_size, conv_hid_size, dilation, conv_dropout)
        self.dropout = nn.Dropout(emb_dropout)
        self.predictor = CoPredictor(label_num, lstm_hid_size, biaffine_size,
                                     conv_hid_size * len(dilation), ffnn_hid_size, out_dropout)

        self.cln = LayerNorm(lstm_hid_size, conditional_size=lstm_hid_size)

    def forward(self, token_ids, pieces2word, dist_inputs, sent_length, grid_mask2d):
        bert_embs = self.bert([token_ids, torch.zeros_like(token_ids)])
        if self.use_bert_last_4_layers:
            bert_embs = torch.stack(bert_embs[-4:], dim=-1).mean(-1)

        length = pieces2word.size(1)
        min_value = torch.min(bert_embs).item()

        # 最大池化
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        # LSTM
        word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=sent_length.max())

        # 条件LayerNorm
        cln = self.cln([word_reps.unsqueeze(2), word_reps])

        # concat
        dis_emb = self.dis_embs(dist_inputs)
        tril_mask = torch.tril(grid_mask2d.clone().long())
        reg_inputs = tril_mask + grid_mask2d.clone().long()
        reg_emb = self.reg_embs(reg_inputs)
        conv_inputs = torch.cat([dis_emb, reg_emb, cln], dim=-1)

        # 卷积层
        conv_inputs = torch.masked_fill(conv_inputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        conv_outputs = self.convLayer(conv_inputs)
        conv_outputs = torch.masked_fill(conv_outputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)

        # 输出层
        outputs = self.predictor(word_reps, word_reps, conv_outputs)
        return outputs

net = Model()