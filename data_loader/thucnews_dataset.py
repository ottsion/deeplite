import shutil
import struct
from collections import defaultdict
from pathlib import Path

import lmdb
import numpy as np
import torch.utils.data
from tqdm import tqdm
import os

"""
 not finished
"""

class ThucnewsDataset(torch.utils.data.Dataset):
    """
    THUCTC: 一个高效的中文文本分类工具
    :param dataset_path: criteo train.txt path.
    :param cache_path: lmdb cache path.
    :param rebuild_cache: If True, lmdb cache is refreshed.
    :param data_sub:  data_sub threshold.
    :param text_len:  text_len threshold.
    Reference:
        https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset
        https://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf
    """

    def __init__(self, dataset_path=None, cache_path='.thucnews', rebuild_cache=False, data_sub=5000, max_text_len=100, min_word_count=2):
        self.data_sub = data_sub
        self.max_text_len = max_text_len
        self.min_word_count = min_word_count
        dirs = os.listdir(dataset_path)
        self.label_code = {label: index for index, label in enumerate(dirs)}
        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('create cache: failed: dataset_path is None')
            self.__build_cache(dataset_path, cache_path)
        print("cache_path: ", cache_path)
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] - 1
            self.vocab_size = np.frombuffer(txn.get(b'vocab_size'), dtype=np.uint32)
            self.gram2_vocab_size = np.frombuffer(txn.get(b'gram2_vocab_size'), dtype=np.uint32)
        print("-vocab_size: ", self.vocab_size)
        print("-gram2_vocab_size: ", self.gram2_vocab_size)

    def __getitem__(self, index):
        print("here..", index)
        with self.env.begin(write=False) as txn:
            print(txn)
            np_array = np.frombuffer(
                txn.get(struct.pack('>I', index)), dtype=np.uint32).astype(dtype=np.long)
        print("np_array; ", np_array)
        return np_array[1:].astype(dtype=np.long), np_array[0].astype(dtype=np.float)

    def __len__(self):
        return self.length

    def __build_cache(self, path, cache_path):
        vocab_mapper, gram2_vocab_mapper = self.__get_feat_mapper(path)
        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            with env.begin(write=True) as txn:
                txn.put(b'vocab_size', self.vocab_size.tobytes())
                txn.put(b'gram2_vocab_size', self.gram2_vocab_size.tobytes())
            for buffer in self.__yield_buffer(path, vocab_mapper, gram2_vocab_mapper):
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        txn.put(key, value)

    def __get_feat_mapper(self, path):
        vocab_cnts = defaultdict(int)
        gram2_vocab_cnts = defaultdict(int)
        dirs = os.listdir(path)
        for dir in dirs:
            sub_path = os.path.join(path, dir)
            file_names = os.listdir(sub_path)
            for file_name in file_names:
                file_path = os.path.join(sub_path, file_name)
                with open(file_path, "r+", encoding="utf-8") as f:
                    pbar = tqdm(f, mininterval=1, smoothing=0.1)
                    pbar.set_description('Create thucnews dataset cache: counting vocab')
                    words = []
                    for line in pbar:
                        words.append(line.rstrip('\n').replace('\t', "").replace(' ', ""))
                    words = "".join(words)
                    for word in words:
                        vocab_cnts[word] += 1
                    for index in range(len(words)-1):
                        if index % 2 == 0:
                            gram2_vocab_cnts[words[index: index+2]] += 1
        vocab_mapper = [word for word, cnt in vocab_cnts.items() if cnt > self.min_word_count and word != " "]
        vocab_mapper = {word: iid+1 for iid, word in enumerate(vocab_mapper)}
        gram2_vocab_mapper = [word for word, cnt in gram2_vocab_cnts.items() if cnt > self.min_word_count and word != " "]
        gram2_vocab_mapper = {word: iid + 1 for iid, word in enumerate(gram2_vocab_mapper)}
        self.vocab_size = np.array([len(vocab_mapper)])
        self.gram2_vocab_size = np.array([len(gram2_vocab_mapper)])
        return vocab_mapper, gram2_vocab_mapper

    def __yield_buffer(self, path, vocab_mapper, gram2_vocab_mapper, buffer_size=int(1e5)):
        item_idx = 0
        buffer = list()
        dirs = os.listdir(path)
        for dir in dirs:
            sub_path = os.path.join(path, dir)
            file_names = os.listdir(sub_path)
            for file_name in file_names:
                file_path = os.path.join(sub_path, file_name)
                with open(file_path, "r+", encoding="utf-8") as f:
                    pbar = tqdm(f, mininterval=1, smoothing=0.1)
                    pbar.set_description('Create huncnews dataset cache: setup lmdb')
                    count = 0
                    words = []
                    for line in pbar:
                        words.append(line.rstrip('\n').replace('\t', ""))
                    words = "".join(words)
                    if self.data_sub != 0 and self.data_sub > count:
                        break
                    print("words:", words)
                    count += 1
                    word_list = []
                    gram_list = []
                    if len(words) > self.max_text_len:
                        words = words[:self.max_text_len+1]
                    else:
                        words = words + "".join([" " for _ in range(self.max_text_len-len(words))])
                    for index in range(len(words)):
                        word_list.append(vocab_mapper.get(words[index], 0))
                        if index % 2 == 0:
                            gram_list.append(gram2_vocab_mapper.get(words[index: index+2], 0))
                    np_array = np.zeros(self.max_text_len * 2 + 1, dtype=np.uint32)
                    np_array[0] = [int(self.label_code[file_name])]
                    for iid, word_index in enumerate(word_list):
                        np_array[iid+1] = word_index
                    for iid, words_index in enumerate(gram_list):
                        np_array[iid+1] = words_index
                    buffer.append((struct.pack('>I', item_idx), np_array.tobytes()))
                    item_idx += 1
                    if item_idx % buffer_size == 0:
                        yield buffer
                        buffer.clear()
                yield buffer
