import nltk
import numpy as np
import pandas as pd

class dataprocessing :
    def __init__(self):
        self.corpus_add = './sample.txt'

    def read_corpus(self):
        # get corpus
        corpus_file = open(self.corpus_add)
        text = corpus_file.readline()
        # tokenize the corpus
        tokens = nltk.word_tokenize(text)
        return tokens

    def build_word_matrix(self, tokens, vector_dim):


        unique_words_n = len(set(tokens))
        # build standard word matrix
        # initialize the elements values
        word_matrix = pd.DataFrame(data = np.random.normal(0,0.5,size = (unique_words_n,vector_dim)),
                                   index = set(tokens), columns=range(vector_dim), dtype=np.float32)

        # truncate outliers
        def truncate(x):
            bound = 0.9
            if x > bound : return bound
            elif x < -1*bound : return -1*bound
            else :return x
        word_matrix = word_matrix.applymap(truncate)

        # Negative samping을 위한 단어별 확률분포 생성
        total_num_words = len(tokens)
        freqdist = nltk.FreqDist(tokens)
        word_temp1 = pd.DataFrame([freqdist])

        word_temp1 = word_temp1/ total_num_words
        word_temp1 = pow(word_temp1, 0.75)
        denom = word_temp1.sum(axis = 1)[0]
        word_prob = word_temp1 / denom   # 클래스 변수에 바로 할당
        word_idx = word_prob.columns
        word_prob = word_prob.values[0].tolist()

        return (word_matrix, unique_words_n, word_idx, word_prob)




