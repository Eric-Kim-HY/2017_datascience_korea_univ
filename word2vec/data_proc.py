import nltk
import numpy as np
import pandas as pd
from numba import jit

class dataprocessing :
    @jit
    def __init__(self):
        self.corpus_add = './text8'
        #self.corpus_add = './sample.txt' # for using sample

        # stopword setting
        self.stpwrds = nltk.corpus.stopwords.words('english')


    def read_corpus(self):
        # get corpus
        corpus_file = open(self.corpus_add)
        text = corpus_file.readline()
        # tokenize the corpus
        tokens = nltk.word_tokenize(text)

        # Stopword 제거
        tokens = [k for k in tokens if k not in self.stpwrds]

        # 숫자제거
        tokens = [k for k in tokens if not k.isdigit()]

        return tokens


    def build_word_matrix(self, tokens, vector_dim):
        # get the number of unique words in the corpus
        unique_words_n = len(set(tokens))
        # build standard word matrix
        # initialize the elements values
        word_matrix = pd.DataFrame(data = np.random.normal(0,0.5,size = (unique_words_n,vector_dim)),
                                   index = set(tokens), columns=range(vector_dim), dtype=np.float32)

        # truncate outliers

        def truncate(x):
            bound = 0.9
            if x > bound : return bound
            elif x < -1 * bound : return -1 * bound
            else : return x
        word_matrix = word_matrix.applymap(truncate)

        # Negative sampling을 위한 단어별 확률분포 생성
        # 해당어 확률 = [해당어 빈도수^(3/4)] / [빈도수^(3/4)의 총합]
        total_num_words = len(tokens)       # 중복 허용한 모든 token 갯수
        freqdist = nltk.FreqDist(tokens)    # corpus 내 token의 빈도수
        word_temp1 = pd.DataFrame([freqdist])

        word_temp1 = word_temp1/ total_num_words
        word_temp1 = pow(word_temp1, 0.75)
        denom = word_temp1.sum(axis = 1)[0] # 모든 word_temp1들의 합을 분모로
        word_prob = word_temp1 / denom   # 클래스 변수에 바로 할당
        word_idx = word_prob.columns
        word_prob = word_prob.values[0].tolist()

        return (word_matrix, unique_words_n, word_idx, word_prob)




