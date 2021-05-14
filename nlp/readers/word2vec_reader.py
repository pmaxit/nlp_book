import math
import random
from collections import Counter

import numpy as np
from numpy.lib.npyio import load
import torch
import torch.optim as optim
from allennlp.common.file_utils import cached_path
from allennlp.data import DataLoader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.training.trainer import GradientDescentTrainer
from overrides import overrides
from scipy.stats import spearmanr
from torch.nn import CosineSimilarity
from torch.nn import functional
from typing import List, Optional
from allennlp.data.fields import TextField, LabelField, ListField
import dill as pickle

np.random.seed(12345)


from itertools import islice

def chunk_list(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())
    
def chunked(file, chunk_size):
    return iter(lambda: file.read(chunk_size), '')


class WordTable:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, input_file_name: str, min_count:int = 10):
        self.negatives = []
        self.discards = []
        self.negpos = 0

        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.input_file_name= input_file_name
        self.read_words(min_count)
        self.initTableNegatives()
        self.initTableDiscards()

    def read_words(self, min_count:int):
        word_frequency = dict()
        for line in open(self.input_file_name, encoding='utf-8'):
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

                    if self.token_count %1000000 == 0:
                        print("Read "+str(int(self.token_count / 1000000)) + "M words")

        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1

        print("Total embeddings: "+ str(len(self.word2id)))

    def initTableDiscards(self):
        t = 0.001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.5
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * WordTable.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [self.id2word[wid]] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)


    def getNegatives(self, target, size):
        response = self.negatives[self.negpos: self.negpos+size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            return np.concatenate((response, self.negatives[0: self.negpos]))
        
        return response

@DatasetReader.register('skip_gram')
class SkipGramReader(DatasetReader):
    def __init__(self, window_size=5, 
                    load_from_pkl:Optional[str]=None,
                    negative_sampling:int=10,
                    chunk_size:int = 65535):
        """ A DatasetReader for reading a plain text corpus and producing instances of skipgram model
            When vocab is not None, this runs sub-sampling of frequent words

            Subsampling is a method of diluting very frequent words, akin to removing stop-words
            THe subsampling method presented in <> randomly removes word that are more frequent than some
            thresold t with a probability of p, where f marks the word's corpus frequency
        """
        super().__init__()
        self.window_size = window_size
        self.load_from_pkl = load_from_pkl
        self.negative_sampling = negative_sampling
        self.chunk_size= chunk_size

    @staticmethod
    def save_to_pkl(obj, file_path:str):
        with open(file_path,'wb') as file:
            pickle.dump(obj, file)

    @staticmethod
    def load_from_pkl(file_path:str):
        with open(file_path,'rb') as file:
            return pickle.load(file)
    
    @overrides
    def _read(self, file_path: str):
        # read the whole dataset first and create freq table with valid words
        if self.load_from_pkl:
            self.word_table = SkipGramReader.load_from_pkl(self.load_from_pkl)
        else:
            self.word_table = WordTable(file_path)
            SkipGramReader.save_to_pkl(self.word_table, self.load_from_pkl)

        with open(cached_path(file_path),'r') as text_file:
            count = 300
            for tokens in chunk_list(text_file.read().split(' '), self.chunk_size):
                if count <= 0:
                    break
                words_not_frequent = [w for w in tokens if w in self.word_table.word2id 
                            and np.random.rand() < self.word_table.discards[self.word_table.word2id[w]]]

                boundary = np.random.randint(1, self.window_size)
                instance_data = [(u,v , self.word_table.getNegatives(v, self.negative_sampling)) for i,u in enumerate(words_not_frequent) 
                            for j,v in enumerate(words_not_frequent[max(i-boundary,0): i+boundary+1]) if u != v]
                for instance in instance_data:
                    token_in = LabelField(instance[0], label_namespace='tags_in')
                    token_out = LabelField(instance[1], label_namespace='tags_in')
                    negatives = ListField([LabelField(w, label_namespace='tags_in') for w in instance[2]])
                    yield Instance({'tokens_in': token_in, 'tokens_out': token_out,'negatives': negatives})

                count -=1