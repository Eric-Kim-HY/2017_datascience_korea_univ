import nltk
import numpy as np
import pandas as pd

class dataprocessing :
    def __init__(self, vector_dim):
        self.corpus_add = './sample.txt'
        pass


    def read_corpus(self):
        # get corpus
        corpus_file = open(self.corpus_add)
        text = corpus_file.readlines()
        # tokenize the corpus
        tokens = nltk.word_tokenize(text)
        return tokens

    def build_word_matrix(self, tokens, vector_dim):

        n_words = len(tokens)
        # build standard word matrix
        word_matrix = pd.DataFrame(index = set(tokens),columns=range(vector_dim), dtype=np.float32)

        TF = nltk.FreqDist(tokens)

        # weight matrix 2개 관리하기 때문에 한번 연산으로 두 개 반환, 단어 빈도수 반환, 단어 종류수 반환
        return (word_matrix, word_matrix)

    # 총단어의 갯수(idx), sampling할 단어 수 n개 입력해 n개의 idx를 받아온다
    def negative_sampling(self, n_unique_words, n_samples):
        return np.random.randint(low = 0, high = n_unique_words, size = n_samples)

