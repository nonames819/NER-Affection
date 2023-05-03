import csv


from bert4torch.tokenizers import Tokenizer
import numpy as np
from bert4torch.snippets import sequence_padding, ListDataset, seed_everything

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
                d = []
                flags = row[2].split(' ')
                for i, flag in enumerate(flags):
                    if flag[0] == 'B':
                        d.append([i, i, flag[2:]])
                    elif flag[0] == 'I':
                        d[-1][1] = i
                if len(sentence) > maxlen - 2:
                    continue
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

                # golden标签
                _grid_labels = np.zeros((length, length), dtype=np.int)
                _grid_mask2d = np.ones((length, length), dtype=np.bool)

                for entity in d:
                    e_start, e_end, e_type = entity[0], entity[1] + 1, entity[-1]

                    if e_end >= maxlen - 2:
                        continue
                    index = list(range(e_start, e_end))
                    for i in range(len(index)):
                        if i + 1 >= len(index):
                            break
                        _grid_labels[index[i], index[i + 1]] = 1
                    _grid_labels[index[-1], index[0]] = categories[e_type]
                _entity_text = set([convert_index_to_text(list(range(e[0], e[1] + 1)), categories[e[-1]]) for e in d])
                D.append((tokens_ids, _pieces2word, _dist_inputs, _grid_labels, _grid_mask2d, _entity_text))
        return D
