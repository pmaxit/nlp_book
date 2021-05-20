import re
import os
import random
import numpy as np

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField
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

logger = logging.getLogger(__name__)    
logging.basicConfig(level=logging.INFO)

ILLEGAL_CHAR_REGEX = "[^a-zA-Z-'.]"
MAX_DICT_WORD_LEN = 20
MIN_DICT_WORD_LEN = 2

def is_alternate_pho_spelling(word):
    return word[-1] == ')' and word[-3] == '(' and word[-2].isdigit()

def should_skip(word):
    if not word[0].isalpha():
        return True
    
    if word[-1] == '.':
        return True

    if re.search(ILLEGAL_CHAR_REGEX, word):
        return True
    
    if len(word) > MAX_DICT_WORD_LEN:
        return True

    if len(word) < MIN_DICT_WORD_LEN:
        return True

    return False

@DatasetReader.register('cmu_reader')
class CMUReader(DatasetReader):
    """ Reads names from dataset file"""
    def __init__(self, source_tokenizer: Tokenizer=None,
            target_tokenizer:Tokenizer=None,
            source_token_indexer: Dict[str,TokenIndexer] = None,
            target_token_indexer: Dict[str,TokenIndexer] = None,
            max_tokens: int = None,
            target_add_start_token:bool=True,
            target_add_end_token:bool=True,
            end_symbol:str=END_SYMBOL,
            start_symbol:str = START_SYMBOL,
            **kwargs) -> None:
        super().__init__(**kwargs)
        self._source_tokenizer = source_tokenizer or CharacterTokenizer(
                    start_tokens=[START_SYMBOL,], 
                    end_tokens=[END_SYMBOL,],
                    lowercase_characters=True)
        self._target_tokenizer = target_tokenizer or WhitespaceTokenizer()
        self._source_token_indexer = source_token_indexer or {'tokens': SingleIdTokenIndexer() }
        self._target_token_indexer = target_token_indexer or {'tokens': SingleIdTokenIndexer() }
        self._max_tokens = max_tokens
        self._start_symbol = start_symbol
        self._end_symbol = end_symbol
        self._target_add_start_token = target_add_start_token
        self._target_add_end_token = target_add_end_token

        # Make sure we add start, end token in target. it will be used in decoding state.
        # Only START_SYMBOL and END_SYMBOl is used 

    @overrides
    def _read(self, file_path: str)->Instance:
        with open(cached_path(file_path),'r') as dict_file:
            for line in dict_file:
                # skip commented lines
                if line[0:3] == ';;;':
                    continue
                
                word, *phonetic = line.strip().split(' ')
                # Alternate pronounciations are formatted : WORD(#) F AH) N EH1 TIHO K"
                # We don't want the (#) considered as part of the word
                if is_alternate_pho_spelling(word):
                    word = word[:word.find('(')]

                if should_skip(word):
                    continue
                if phonetic is None or len(phonetic) <= 0:
                    print(line, phonetic)
                    print("ERROR phonetic is None")
                else:
                    yield self.text_to_instance(word, ' '.join(phonetic))


    @overrides
    def text_to_instance(self, word:str, phonetic:Optional[str]= None) -> Instance:
        word_tokenized = self._source_tokenizer.tokenize(word)
        if len(word_tokenized) > self._max_tokens:
            word_tokenized = word_tokenized[:self._max_tokens]
        
        source_field = TextField(word_tokenized, self._source_token_indexer)

        if phonetic:
            phonetic_tokenized = self._target_tokenizer.tokenize(phonetic)
            if len(phonetic_tokenized) > self._max_tokens:
                phonetic_tokenized = phonetic_tokenized[:self._max_tokens]

            if self._target_add_start_token:
                phonetic_tokenized.insert(0, Token(copy.deepcopy(self._start_symbol)))
            
            if self._target_add_end_token:
                phonetic_tokenized.append(Token(copy.deepcopy(self._end_symbol)))
            
            target_field = TextField(phonetic_tokenized, self._target_token_indexer)
            return Instance({'source_tokens': source_field, 'target_tokens': target_field})

        else:
            return Instance({'source_tokens': source_field})

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["source_tokens"]._token_indexers = self._source_token_indexer  # type: ignore
        if "target_tokens" in instance.fields:
            instance.fields["target_tokens"]._token_indexers = self._target_token_indexer  # type: ignore