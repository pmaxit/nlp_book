from typing import Optional, Dict
import torch

from allennlp.common import Params
from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, FeedForward
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.regularizers import RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import BooleanAccuracy

import sys
sys.path.insert(0, './')

from nlp.metrics.multilabel_f1 import MultiLabelF1Measure

@Model.register('toxic')
class ToxicModel(Model):
    def __init__(self, 
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            encoder : Seq2VecEncoder,
            classifier_feedforward: FeedForward,
            initializer: InitializerApplicator = InitializerApplicator(),
            regularizer: Optional[RegularizerApplicator] = None)->None:

        super().__init__(vocab)
        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size('labels')
        self.encoder = encoder
        self.classifier_feedforward = classifier_feedforward
        self.loss = torch.nn.MultiLabelMarginLoss()
        self.f1 = MultiLabelF1Measure()

        initializer(self)

    def get_metrics(self, reset: bool = False)->Dict[str, float]:
        precision, recall , f1, accuracy = self.f1.get_metric(reset)
        return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}

    def forward(self, 
            text: Dict[str, torch.Tensor],
            labels: torch.LongTensor = None)->Dict[str, torch.Tensor]:

        embedded_text = self.text_field_embedder(text)
        mask = util.get_text_field_mask(text)
        encoded_text = self.encoder(embedded_text, mask)

        logits = self.classifier_feedforward(encoded_text)
        probabilities = torch.nn.functional.sigmoid(logits)

        output_dict = {'logits': logits, 'probabilities': probabilities}

        if labels is not None:
            loss = self.loss(logits, labels.squeeze(-1).long())
            output_dict['loss'] = loss

            predictions = (logits.data > 0.0).long()
            label_data = labels.squeeze(-1).data.long()
            self.f1(predictions, label_data)
            
        return output_dict

