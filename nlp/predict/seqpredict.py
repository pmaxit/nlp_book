from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp_models.generation.predictors.seq2seq import Seq2SeqPredictor
from pprint import pprint

@Predictor.register('my_seq2seq')
class MySeq2SeqPredictor(Seq2SeqPredictor):

    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)

        outputs['meta'] = {}

        for k, v in instance.fields.items():
            outputs['meta'][k] = v.tokens

        return sanitize(outputs)