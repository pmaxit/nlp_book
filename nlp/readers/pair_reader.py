import re
import os
import random
import numpy as np

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, TensorField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer,TokenCharactersIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer, WhitespaceTokenizer, CharacterTokenizer
from allennlp.common.util import START_SYMBOL, END_SYMBOL

import itertools
from typing import Dict, List, cast, Optional
from allennlp.common.util import ensure_list
import logging
from allennlp.data import Vocabulary
from overrides import overrides
from allennlp.common.file_utils import cached_path
import copy
import csv

logger = logging.getLogger(__name__)    
logging.basicConfig(level=logging.INFO)

@DatasetReader.register('pair_reader')
class pair_reader(DatasetReader):
    """ Reads names from dataset file"""
    def __init__(self, 
            tokenizer: Tokenizer,
            token_indexers: Dict[str, TokenIndexer],
            combine_input_fields : Optional[bool] = False,
            add_special_symbols:Optional[bool]=False,
            **kwargs) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer or CharacterTokenizer(lowercase_characters=True)
        self._token_indexer = token_indexers or {'tokens': SingleIdTokenIndexer() }
        self._combine_input_fields = combine_input_fields
        self._add_special_symbols = add_special_symbols

        self._start_symbol = START_SYMBOL
        self._end_symbol = END_SYMBOL

    @overrides
    def _read(self, file_path: str)->Instance:
        logger.info("Reading instances from lines in file at %s", file_path)
        with open(cached_path(file_path),'r') as data_file:
            csv_in = csv.reader(data_file)
            next(csv_in)
            for row in csv_in:
                if len(row) >=5:
                    yield self.text_to_instance(word1=row[0], word2=row[1], score=row[5])

    @overrides
    def text_to_instance(self, word1:str, word2:str, score:Optional[float]=None) -> Instance:
        fields: Dict[str, Field] = {}

        def add_special_tokens(tokens):
            # add special token
            if self._add_special_symbols:
                tokens.insert(0, Token(copy.deepcopy(self._start_symbol)))
                tokens.append(Token(copy.deepcopy(self._end_symbol)))
            return tokens

        word1 = self._tokenizer.tokenize(word1)
        word2 = self._tokenizer.tokenize(word2) 

        if self._combine_input_fields:
            # this will be required for encoder type architecture
            raise NotImplementedError
        else:
            word1_tokens = add_special_tokens(word1)
            word2_tokens = add_special_tokens(word2)
            fields['word1'] = TextField(word1_tokens, self._token_indexer)
            fields['word2'] = TextField(word2_tokens, self._token_indexer)

        if score is not None:
            score = np.array(float(score))
            score = score.astype('double')
            fields['score'] = TensorField(score)
        return Instance(fields) 

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["word1"]._token_indexers = self._token_indexer  # type: ignore
        if "word2" in instance.fields:
            instance.fields["word2"]._token_indexers = self._token_indexer  # type: ignore