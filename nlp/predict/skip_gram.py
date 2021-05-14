from typing import List

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from torch.nn import CosineSimilarity
from torch.nn import functional
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.token_embedders import Embedding
from collections import Counter
from allennlp.models.archival import load_archive

from overrides import overrides

import sys
sys.path.insert(0,'/Users/puneet/Projects/pytorch/practice/book_nlp')

from nlp.readers.word2vec_reader import SkipGramReader
from nlp.models.skip_gram import SkipGramModel

def write_embeddings(embedding: Embedding, file_path, vocab:Vocabulary):
    with open(file_path, mode='w') as f:
        for index, token in vocab.get_index_to_token_vocabulary('tags_in').items():
            values = ['{:.5f}'.format(val) for val in embedding.weight[index]]
            f.write(' '.join([token]+values))
            f.write('\n')

def get_synonyms(token:str, embedding:Model, vocab:Vocabulary, num_synonyms:int = 10):
    "Given a token, return a list of top N most similar words to the token"
    token_id = vocab.get_token_index(token, 'tags_in')
    token_vec = embedding.weight[token_id]

    cosine = CosineSimilarity(dim=0)
    sims = Counter()

    for index, token in vocab.get_index_to_token_vocabulary('tags_in').items():
        sim = cosine(token_vec, embedding.weight[index]).item()
        sims[token] = sim

    return sims.most_common(num_synonyms)

def plot_embedding(model):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    # getting embeddings from embedding layer of our model, by name
    embeddings = model.embedding_in.weight.to('cpu').data.numpy()
    vocab  = model.vocab

    viz_words = 380
    tsne = TSNE()
    embed_tsne = tsne.fit_transform(embeddings[:viz_words,:])

    fig, ax  = plt.subplots(figsize=[16, 16])
    for idx in range(viz_words):
        plt.scatter(*embed_tsne[idx, :], color='steelblue')
        plt.annotate(vocab.get_token_from_index(idx), (embed_tsne[idx,0], embed_tsne[idx,1], embed_tsne[idx, 1]), alpha=0.7)

    plt.show()
    
def main(archive_file:str):
    archive = load_archive(archive_file)
    predictor = Predictor.from_archive(archive)

    embedding = predictor._model.embedding_in
    vocab = predictor._model.vocab

    #write_embeddings(embedding, "./junks/text8_emb.txt", vocab)

    print(get_synonyms('one', embedding, vocab))
    print(get_synonyms('december', embedding, vocab))
    print(get_synonyms('flower', embedding, vocab))
    print(get_synonyms('design', embedding, vocab))
    print(get_synonyms('snow', embedding, vocab))

if __name__ == '__main__':
    main(sys.argv[1])
    

