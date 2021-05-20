
from typing import Dict, Optional
from allennlp.nn.regularizers import RegularizerApplicator

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder, FeedForward
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy

@Model.register('seq_classifier')
class BasicClassifier(Model):
    def __init__(self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            seq2vec_encoder: Seq2VecEncoder,
            dropout:float=0,
            label_namespace:str='label',
            initializer: InitializerApplicator= InitializerApplicator(),
            regularizer: Optional[RegularizerApplicator] = None,
            feedforward: Optional[FeedForward] = None,
            )->None:

        super().__init__(vocab, regularizer)
        self._text_field_embedder=text_field_embedder
        self._seq2vec_encoder=seq2vec_encoder
        self._feedforward = feedforward
        
        if feedforward is not None:
            self._classifier_input_dim= feedforward.get_output_dim()
        else:
            self._classifier_input_dim = seq2vec_encoder.get_output_dim()
        self.bn = nn.BatchNorm1d(num_features=self._classifier_input_dim)
        if dropout:
            self._dropout = nn.Dropout(dropout)
        else:
            self._dropout = None

        self._num_labels = vocab.get_vocab_size(namespace=label_namespace)
        self._classification_layer = nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._loss = nn.CrossEntropyLoss()

        initializer(self)
    
    def forward(self,
        tokens: Dict[str, torch.LongTensor],
        label: torch.IntTensor=None)-> Dict[str, torch.Tensor]:
        embedded_text= self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).float()

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)
        
        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)
        
        embedded_text = self.bn(embedded_text)
        
        logits = self._classification_layer(embedded_text)
        probs = F.softmax(logits, dim=1)

        output_dict = {
            'logits': logits,
            'probs': probs
        }
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict['loss']=loss
            self._accuracy(logits, label)

        return output_dict

    def get_metrics(self, reset:bool=False)->Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}