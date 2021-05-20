import csv
from typing import Dict, Optional
from allennlp.data.fields.field import Field
from allennlp.data.vocabulary import Vocabulary

from overrides import overrides
import os.path as osp
import numpy as np

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.common.util import ensure_list
from allennlp.common.file_utils import cached_path

from itertools import chain
import tarfile
from pathlib import Path

@DatasetReader.register('my_imdb')
class IMDBDatasetReader(DatasetReader):

    TAR_URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    TRAIN_DIR = 'aclImdb/train'
    TEST_DIR = 'aclImdb/test'

    def __init__(self, 
            token_indexers: Dict[str, TokenIndexer] = None,
            tokenizer: Tokenizer = None
        ):
        super().__init__()
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path:str):
        tar_path = cached_path(self.TAR_URL)
        tf = tarfile.open(tar_path, 'r')
        cache_dir = Path(osp.dirname(tar_path))

        if not (cache_dir / self.TRAIN_DIR).exists() and not ( cache_dir / self.TEST_DIR).exists():
            tf.extractall(cache_dir)
        
        if file_path == 'train':
            pos_dir = osp.join(self.TRAIN_DIR, 'pos')
            neg_dir = osp.join(self.TRAIN_DIR, 'neg')

        elif file_path == 'test':
            pos_dir = osp.join(self.TEST_DIR, 'pos')
            neg_dir = osp.join(self.TEST_DIR,'neg')
        else:
            raise ValueError(f"only train and test are valid file_path but {file_path} is given !")

        path = chain(Path(cache_dir.joinpath(pos_dir)).glob('*.txt'),
                    Path(cache_dir.joinpath(neg_dir)).glob('*.txt'))
        for p in path:
            yield self.text_to_instance(p.read_text(), 0 if 'pos' in str(p) else 1)
        
    @overrides
    def text_to_instance(self, string: str, label:Optional[int]=None)-> Instance:
        fields :Dict[str, Field]= {}
        tokens = self._tokenizer.tokenize(string)
        fields['tokens'] = TextField(tokens, self._token_indexers)
        
        if label is not None:
            fields['label']=  LabelField(label,skip_indexing=True)
        
        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["tokens"]._token_indexers = self._token_indexers  # type: ignore

if __name__ == '__main__':
    reader = IMDBDatasetReader()
    instances = ensure_list(reader.read('train'))
    vocab = Vocabulary.from_instances(instances)
    print(vocab)
