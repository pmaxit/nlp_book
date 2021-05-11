  
from typing import Iterable, Dict, Tuple, List
from typing import Iterable, Dict, Tuple, List

import numpy as np
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, ArrayField, MetadataField, NamespaceSwappingField

from allennlp.data import Vocabulary

import datasets

@DatasetReader.register('cnn_dailymail')
class CNNDailyMailReader(DatasetReader):
    def __init__(self, 
            dataset_name:str ='cnn_dailymail',
            tokenizer: Tokenizer = None,
            source_token_indexers: Dict[str, TokenIndexer] = None,
            target_token_indexers: Dict[str, TokenIndexer] = None,
            source_max_tokens: int =400,
            target_max_tokens: int = 100,
            seperate_namespaces: bool =False,
            target_namespace: str="target_tokens",
            save_copy_fields: bool = False,
            save_pgn_fields: bool = False,
            lowercase: bool=True):
        super().__init__()
        self._lowercase = lowercase
        self._dataset_name = dataset_name
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens

        self._tokenizer = tokenizer or WhitespaceTokenizer()

        tokens_indexer = {'tokens': SingleIdTokenIndexer(lowercase_tokens=lowercase)}
        self._source_token_indexers = source_token_indexers or tokens_indexer
        self._target_token_indexers = target_token_indexers or tokens_indexer
        self._save_copy_fields = save_copy_fields
        self._save_pgn_fields = save_pgn_fields

        self._target_namespace = 'tokens'
        self.train = datasets.load_dataset(dataset_name,'3.0.0',split='train',)
        self.val = datasets.load_dataset(dataset_name,'3.0.0', split='validation[:5%]')

        if seperate_namespaces:
            self._target_namespace = target_namespace
            second_tokens_indexer = {'tokens': SingleIdTokenIndexer(namespace = target_namespace)}
            self._target_token_indexers = target_token_indexers or second_tokens_indexer

    def _read(self, mode:str)-> Iterable[Instance]:

        if mode == 'train':
            for i in range(1000):
                ex = self.train[i]
                yield self.text_to_instance(ex['article'],ex['highlights'])
        else:
            print("length of val ", len(self.val))
            for i in range(len(self.val)):
                ex = self.val[i]
                yield self.text_to_instance(ex['article'],ex['highlights'])
        
    @staticmethod
    def _tokens_to_ids(tokens: List[Token], lowercase=True) -> List[int]:
        ids = dict()
        out = list()
        for token in tokens:
            token_text = token.text.lower() if lowercase else token.text
            out.append(ids.setdefault(token_text, len(ids)))
        return out

    def text_to_instance(self, source: str, target: str = None) -> Instance:
        def prepare_text(text, max_tokens):
            text = text.lower() if self._lowercase else text
            tokens = self._tokenizer.tokenize(text)[:max_tokens]
            tokens.insert(0, Token(START_SYMBOL))
            tokens.append(Token(END_SYMBOL))
            return tokens

        source_tokens = prepare_text(source, self._source_max_tokens)
        source_tokens_indexed = TextField(source_tokens, self._source_token_indexers)
        result = {'source_tokens': source_tokens_indexed}
        meta_fields = {}

        if self._save_copy_fields:
            source_to_target_field = NamespaceSwappingField(source_tokens[1:-1], self._target_namespace)
            result["source_to_target"] = source_to_target_field
            meta_fields["source_tokens"] = [x.text for x in source_tokens[1:-1]]

        if self._save_pgn_fields:
            source_to_target_field = NamespaceSwappingField(source_tokens, self._target_namespace)
            result["source_to_target"] = source_to_target_field
            meta_fields["source_tokens"] = [x.text for x in source_tokens]

        if target:
            target_tokens = prepare_text(target, self._target_max_tokens)
            target_tokens_indexed = TextField(target_tokens, self._target_token_indexers)
            result['target_tokens'] = target_tokens_indexed

            if self._save_pgn_fields:
                meta_fields["target_tokens"] = [y.text for y in target_tokens]
                source_and_target_token_ids = self._tokens_to_ids(source_tokens + target_tokens, self._lowercase)
                source_token_ids = source_and_target_token_ids[:len(source_tokens)]
                result["source_token_ids"] = ArrayField(np.array(source_token_ids, dtype='long'))
                target_token_ids = source_and_target_token_ids[len(source_tokens):]
                result["target_token_ids"] = ArrayField(np.array(target_token_ids, dtype='long'))

            if self._save_copy_fields:
                meta_fields["target_tokens"] = [y.text for y in target_tokens[1:-1]]
                source_and_target_token_ids = self._tokens_to_ids(source_tokens[1:-1] + target_tokens, self._lowercase)
                source_token_ids = source_and_target_token_ids[:len(source_tokens)-2]
                result["source_token_ids"] = ArrayField(np.array(source_token_ids))
                target_token_ids = source_and_target_token_ids[len(source_tokens)-2:]
                result["target_token_ids"] = ArrayField(np.array(target_token_ids))

        elif self._save_copy_fields:
            source_token_ids = self._tokens_to_ids(source_tokens[1:-1], self._lowercase)
            result["source_token_ids"] = ArrayField(np.array(source_token_ids))
        elif self._save_pgn_fields:
            source_token_ids = self._tokens_to_ids(source_tokens, self._lowercase)
            result["source_token_ids"] = ArrayField(np.array(source_token_ids))
        if self._save_copy_fields or self._save_pgn_fields:
            result["metadata"] = MetadataField(meta_fields)
        
        return Instance(result)
