import os
import json
import torch
import random
import numpy as np


def set_random_seed(seed=None, gpu=False):
    seed = np.random.randint(int(1e6)) if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if gpu:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
    return seed


#Script Load
def load_json_scripts(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        return json_data

def load_label_json(labels_path):
    with open(labels_path, encoding="utf-8") as label_file:
        labels = json.load(label_file)
        char2index = dict()
        index2char = dict()

        for index, char in enumerate(labels):
            char2index[char] = index
            index2char[index] = char

        return char2index, index2char


def load_label_index(label_path):
    char2index = dict()  # [ch] = id
    index2char = dict()  # [id] = ch
    print(label_path)
    with open(label_path, 'r', encoding="utf-8") as f:
        for no, line in enumerate(f):
            if line[0] == '#':
                continue

            index, char, freq = line.strip().split('\t')
            char = char.strip()
            if len(char) == 0:
                char = ' '

            char2index[char] = int(index)
            index2char[int(index)] = char

    return char2index, index2char

def save2json(file_nm, df_dict):
    with open(file_nm, 'w', encoding='utf-8') as fp:
        json.dump(df_dict, fp, ensure_ascii=False, indent=4, sort_keys=True)

    print("save completed")

def init_report_dict(key_list):
    report_dict = {}
    for key in key_list:
        report_dict[key] = []
    return report_dict

def tensor2number(tensor):
    item = tensor.cpu().data.numpy().item() if tensor else 0.
    return item

class TrainReport(object):
    def __init__(self, report_dict):
        self.report_dict = report_dict

    def update_report_dict(self, result):
        for key in self.report_dict:
            self.report_dict[key] = tensor2number(result[key])

    def append_batch_result(self, batch_size, result):
        for key in self.report_dict:
            self.report_dict[key].append(tensor2number(result[key])*batch_size)
            # self.report_dict[key].append(result[key] * batch_size)

    def compute_mean(self, num_examples):
        for key in self.report_dict:
            self.report_dict[key] = np.sum(self.report_dict[key]) / num_examples

    def reset_dict(self):
        for key in self.report_dict:
            self.report_dict[key] = []

    def result_str(self):
        str = '  '.join(f"{metric}: {v:.5f}" for metric, v in self.report_dict.items())
        return str

class ASRTrainReport(object):
    def __init__(self, report_dict):
        self.report_dict = report_dict

    def update_report_dict(self, result):
        for key in self.report_dict:
            self.report_dict[key] = tensor2number(result[key])

    def append_batch_result(self, batch_size, result):
        for key in self.report_dict:
            self.report_dict[key].append(tensor2number(result[key])*batch_size)
            # self.report_dict[key].append(result[key] * batch_size)

    def compute_mean(self, num_examples):
        for key in self.report_dict:
            self.report_dict[key] = np.sum(self.report_dict[key]) / num_examples

    def reset_dict(self):
        for key in self.report_dict:
            self.report_dict[key] = []

    def result_str(self):
        str = '  '.join(f"{metric}: {v:.5f}" for metric, v in self.report_dict.items())
        return str

# dot notation access
class DotConfig:
    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, k):
        v = self._cfg[k]
        if isinstance(v, dict):
            return DotConfig(v)
        return v