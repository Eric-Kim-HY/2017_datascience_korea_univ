import nltk
import numpy as np
import pandas as pd

class dataprocessing :
    def __init__(self, vector_dim):
        self.corpus_add = './sample.txt'
        self.vector_dim = vector_dim
        pass


    def read_corpus(self):
        # get corpus
        corpus_file = open(self.corpus_add)
        text = corpus_file.readlines()
        # tokenize the corpus
        tokens = nltk.word_tokenize(text)
        return tokens

    def build_word_matrix(self, corpus):
        tokens = self.read_corpus()

        # build standard word matrix
        word_matrix = pd.DataFrame(index = set(tokens),columns=range(self.vector_dim), dtype=np.float32)

        TF = nltk.FreqDist(tokens)

        n_words = word_matrix.shape[0]

        # weight matrix 2개 관리하기 때문에 한번 연산으로 두 개 반환, 단어 빈도수 반환, 단어 종류수 반환
        return (word_matrix, word_matrix, TF, n_words)

    def negative_sampling(self, sth):
        pass

