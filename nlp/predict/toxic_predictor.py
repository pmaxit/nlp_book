from typing import List

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor
from overrides import overrides

@Predictor.register('simple_pred')
class ToxicPredictor(Predictor):
    
    def predict(self, word1: str, word2:str)-> JsonDict:
        return self.predict_json({'word1': word1, 'word2': word2})
    
    @overrides
    def _json_to_instance(self, json_dict: JsonDict)->Instance:
        word1 = json_dict['word1']
        word2 = json_dict['word2']
        return self._dataset_reader.text_to_instance(word1= word1, word2=word2)
    