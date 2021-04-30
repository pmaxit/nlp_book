import csv
from typing import Dict, Optional
import logging
import copy

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, Token,WhitespaceTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)

@DatasetReader.register("seq2seq_rev")
class Seq2SeqDatasetReader(DatasetReader):
    def __init__(self,
        source_tokenizer: Tokenizer=None,
        target_tokenizer: Tokenizer=None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        **kwargs
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self._source_tokenzier = source_tokenizer or WhitespaceTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenzier
        self.source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}

        self._target_token_indexers = target_token_indexers or self.source_token_indexers


    def _read(self, file_path: str):
        with open(cached_path(file_path),'r') as data_file:
            for line_num, row in self.shard_iterable(enumerate(csv.reader(data_file, delimiter='\t'))):
                source_sequence , target_sequence = row
                yield self.text_to_instance(source_sequence, target_sequence)

    def text_to_instance(self, source_string: str, target_string = None)-> Instance:
        tokenized_source = self._source_tokenzier.tokenize(source_string)
        source_field = TextField(tokenized_source, self.source_token_indexers)

        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            # Need to have a token to start decoding
            tokenized_target.insert(0, Token(START_SYMBOL))

            # And also that decoding has stopped
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target,  self._target_token_indexers)

            return Instance({'source_tokens': source_field, 'target_tokens': target_field})

        else:
            return Instance({'source_tokens': source_field})

    @overrides
    def apply_token_indexers(self, instance:Instance)->None:
        instance.fields['source_tokens']._token_indexers = self.source_token_indexers
        instance.fields['target_tokens']._token_indexers = self._target_token_indexers

if __name__ == '__main__':
    dataset_reader = Seq2SeqDatasetReader(
        source_token_indexers={"tokens":  SingleIdTokenIndexer(namespace='source_tokens')},
        target_token_indexers={"tokens": SingleIdTokenIndexer(namespace='target_tokens')}
    )

    instances = list(dataset_reader.read('./data/reverse/train.csv'))
    print(instances[0])
