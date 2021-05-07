from typing import List

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor
from overrides import overrides

@Predictor.register('my_toxic')
class ToxicPredictor(Predictor):
    
    def predict(self, sentence: str)-> JsonDict:
        return self.predict_json({'sentence': sentence})
    
    @overrides
    def _json_to_instance(self, json_dict: JsonDict)->Instance:
        sentence = json_dict['sentence']
        return self._dataset_reader.text_to_instance(sentence)
    