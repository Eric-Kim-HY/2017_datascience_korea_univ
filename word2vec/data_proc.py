import nltk
import numpy as np
import pandas as pd

class dataprocessing :
    def __init__(self, vector_dim):
        self.stem = nltk.stem.porter.PorterStemmer()
        self.corpus_add = './sample.txt'
        self.vector_dim = vector_dim
        pass

    def build_word_matrix(self, corpus):
        # get corpus
        corpus_file = open(self.corpus_add)
        text = corpus_file.readlines()
        # tokenize the corpus
        tokens =  nltk.word_tokenize(text)

        # build standard word matrix
        word_matrix = pd.DataFrame(index = set(tokens),columns=range(self.vector_dim), dtype=np.float32)



        # weight matrix 2개 관리하기 때문에 한번 연산으로 두 개 반환
        return (word_matrix, word_matrix)
