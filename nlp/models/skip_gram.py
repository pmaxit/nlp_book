from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
import numpy as np
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
from allennlp.modules.text_field_embedders import TextFieldEmbedder
import torch.nn.functional as F

import math
import random
from collections import Counter
from torch.nn import init
import torch.nn as nn

@Model.register('skipgram_simple2')
class SkipGramModel(Model):
    def __init__(self, vocab, embedding_in: Embedding, embedding_out:Embedding=None, cuda_device=-1):
        super().__init__(vocab)
        self.embedding_in = embedding_in
        self.embedding_out = embedding_out
        
        
        self.emb_dimension = self.vocab.get_vocab_size('tags_in')
        self.embedding_in_bias = nn.Parameter(torch.tensor(1.))
        self.embedding_out_bias = nn.Parameter(torch.tensor(1.))
        self.init_emb()

    def init_emb(self):
        
        initrange = 0.5/self.emb_dimension
        
        init.uniform_(self.embedding_in.weight.data, -initrange, initrange)
        init.uniform_(self.embedding_out.weight.data, -initrange, initrange)
        init.constant_(self.embedding_out.weight.data, 0)

    def forward(self, tokens_in, tokens_out, negatives):
        # calculate loss for positive examples
        embedded_in = self.embedding_in(tokens_in) + self.embedding_in_bias
        embedded_out = self.embedding_out(tokens_out) + self.embedding_out_bias
        embedded_neg = self.embedding_out(negatives) + self.embedding_out_bias

        score = torch.mul(embedded_in, embedded_out).sum(dim=1)
        score = F.logsigmoid(score)

        neg_score = torch.bmm(embedded_neg, embedded_in.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1*neg_score).sum(dim=1)

        total_score = 0.3*score + 0.7*neg_score
        #total_score = score
        return {'loss': -total_score.sum()}


@Model.register('skipgram_simple3')
class SkipGramModel2(Model):
    def __init__(self, vocab, embedding_in: Embedding, embedding_out:Embedding=None, cuda_device=-1,dropout:float=0.3):
        super().__init__(vocab)
        self.embedding_in = embedding_in        
        self.embedding_out = embedding_out
        self.emb_dimension = self.vocab.get_vocab_size('tags_in')

        self.linear1 = nn.Linear(in_features=embedding_in.get_output_dim(), out_features= self.emb_dimension)        
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(self.emb_dimension)
        self.init_emb()

    def init_emb(self):
        
        initrange = 0.5/self.emb_dimension
        
        init.uniform_(self.embedding_in.weight.data, -initrange, initrange)
        init.uniform_(self.embedding_out.weight.data, -initrange, initrange)
        init.constant_(self.embedding_out.weight.data, 0)

    def forward(self, tokens_in, tokens_out, negatives):
        # calculate loss for positive examples
        embedded_in = self.embedding_in(tokens_in)
        embedded_in = self.dropout(embedded_in)
        logits = self.bn(self.linear1(embedded_in))

        # get positive score
        score = torch.gather(logits, dim=-1, index = tokens_out.unsqueeze(1)).squeeze(1)
        score = F.logsigmoid(score).sum(dim=-1)
        # get negative score
        neg_score = torch.gather(logits, dim=-1, index = negatives).sum(dim=1)
        neg_score = F.logsigmoid(-1*neg_score).sum(dim=-1)

        total_score = score + neg_score
        return {'loss': -total_score.sum()}


#@Model.register('skipgram_neg')
class SkipGramNegativeSamplingModel(Model):
    def __init__(self, vocab: Vocabulary, embedding_in: Embedding, embedding_out: Embedding, neg_samples=10, cuda_device=-1):
        super().__init__(vocab)
        self.embedding_in = embedding_in
        self.embedding_out = embedding_out
        self.neg_samples = neg_samples
        self.cuda_device = cuda_device

        check_if_counter = getattr(vocab, '_retained_counter', None)
        if not check_if_counter:
            return
        # pre-compute probability for negative sampling
        token_to_probs = {}
        token_counts = vocab._retained_counter['tags_in']
        assert len(token_counts) > 2

        total_counts = sum(token_counts.values())
        total_probs = 0.

        for token, counts in token_counts.items():
            unigram_freq = counts / total_counts
            unigram_freq = math.pow(unigram_freq, 3 / 4)
            token_to_probs[token] = unigram_freq
            total_probs += unigram_freq

        self.neg_sample_probs = np.ndarray((vocab.get_vocab_size('tags_in'),))
        for token_id , token in vocab.get_index_to_token_vocabulary('tags_in').items():
            self.neg_sample_probs[token_id] = token_to_probs.get(token, 0) / total_probs
        
    def forward(self, tokens_in , tokens_out, negatives=None):
        batch_size = tokens_out.shape[0]

        # calculate loss for positive examples
        embedded_in = self.embedding_in(tokens_in)
        embedded_out = self.embedding_out(tokens_out)
        inner_positive = torch.mul(embedded_in, embedded_out).sum(dim=1)
        log_prob = functional.logsigmoid(inner_positive)

        # Generate negative samples
        negative_out = np.random.choice(a=self.vocab.get_vocab_size('tags_in'), size=batch_size*self.neg_samples, p = self.neg_sample_probs)

        negative_out = torch.LongTensor(negative_out).view(batch_size, self.neg_samples)
        if self.cuda_device > -1:
            negative_out = negative_out.to(self.cuda_device)

        # subtract loss for negative examples
        embedded_negative_out = self.embedding_out(negative_out)
        inner_negative = torch.bmm(embedded_negative_out, embedded_in.unsqueeze(2)).squeeze()
        log_prob += functional.logsigmoid(-1 * inner_negative).sum(dim=1)

        return {'loss': -log_prob.mean()}