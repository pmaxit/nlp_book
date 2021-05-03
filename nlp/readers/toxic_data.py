from typing import List, Dict, Iterable
import csv
import sys
import re

import tqdm
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, LabelField, ListField
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from nltk.corpus import stopwords
from allennlp.data import Vocabulary
from allennlp.data.batch import Batch
from allennlp.common.util import ensure_list

def clean_text(text, remove_stopwords=True):
    output = ""
    text = str(text).replace("\n","")
    text = re.sub(r'[^\w\s]', '',text).lower()
    if remove_stopwords:
        text = text.split(" ")
        for word in text:
            if word not in stopwords.words('english'):
                output = output + " "+ word
    else:
        output = text
    return str(output.strip())[1:-3].replace("  "," ")

@DatasetReader.register('toxic')
class ToxicReader(DatasetReader):
    """ Toxic dataset """
    def __init__(self, max_length: int = None, tokenizer: Tokenizer = None,
                token_indexers: Dict[str, TokenIndexer] = None,
                fill_in_empty_labels: bool = False) -> None:
        super().__init__()
        self.max_length = max_length
        self.fill_in_empty_labels = fill_in_empty_labels
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexer = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def _read(self, file_path: str,skip_header:bool =True)->Iterable[Instance]:
        with open(file_path, 'r') as data_file:
            reader = csv.reader(data_file)
            for row in tqdm.tqdm(reader):
                _, text, *labels = row
                if skip_header:
                    skip_header = False
                    continue
                yield self.text_to_instance(text, labels)

    def text_to_instance(self, 
                        text: str,
                        labels: List[str] = None)->Instance:
        # first clean text
        text = clean_text(text)

        if self.max_length is not None:
            text = text[:self.max_length]
        
        tokenized_text = self._tokenizer.tokenize(text)
        text_field = TextField(tokenized_text, self._token_indexer)
        fields = {'text': text_field}
        if labels or self.fill_in_empty_labels:
            labels = labels or [0, 0, 0, 0, 0, 0]

            toxic, severe_toxic, obscene , threat, insult, identity_hate = labels
            fields['labels'] = ListField([
                LabelField(int(toxic), skip_indexing=True),
                LabelField(int(severe_toxic), skip_indexing=True),
                LabelField(int(obscene), skip_indexing=True),
                LabelField(int(threat), skip_indexing=True),
                LabelField(int(insult), skip_indexing=True),
                LabelField(int(identity_hate), skip_indexing=True)
            ])

            return Instance(fields)

def setup_model(params_file, dataset_file):
    params = Params.from_file(params_file)

    reader = DatasetReader.from_params(params['dataset_reader'])
    instances = ensure_list(reader.read(str(dataset_file)))

    if 'vocabulary' in params:
        vocab_params = params['vocabulary']
        vocab = Vocabulary.from_params(params=vocab_params, instances=instances)
    else:
        vocab = Vocabulary.from_instances(instances)
    
    vocab.save_to_files("new_vocab")
    dataset = Batch(instances)
    dataset.index_instances(vocab)
    
    print(dataset.as_tensor_dict())

if __name__ == '__main__':
    import sys
    setup_model(sys.argv[1], sys.argv[2])