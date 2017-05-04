import nltk
import numpy as np
import pandas as pd

class dataprocessing :
    def __init__(self, vector_dim):
        self.corpus_add = './sample.txt'
        self.unique_words_n = None # 총 단어 종류 수
        self.word_idx = None # 단어 확률분포에 대응되는 idx word
        self.word_prob = None  # 단어 확률 분포 변수( list 할당 예정 )
        pass


    def read_corpus(self):
        # get corpus
        corpus_file = open(self.corpus_add)
        text = corpus_file.readline()
        # tokenize the corpus
        tokens = nltk.word_tokenize(text)
        return tokens

    def build_word_matrix(self, tokens, vector_dim):

        self.unique_words_n = len(set(tokens))

        # build standard word matrix
        # initialize the elements values
        word_matrix = pd.DataFrame(data = np.random.normal(0,0.5,size = (self.unique_words_n,vector_dim)),
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
        word_temp1 = pd.DataFrame([freqdist]) / total_num_words
        word_temp1 = pow(word_temp1, 0.75)
        denom = word_temp1.sum(axis = 1)[0]
        word_prob = word_temp1 / denom   # 클래스 변수에 바로 할당
        self.word_idx = word_prob.columns
        self.word_prob = word_prob.values[0].tolist()

        return word_matrix


    # sampling할 단어 수 n개 입력해 n개의 idx words 를 받아온다
    def negative_sampling(self, n_samples):


        # word_prob 확률을 이용해서 (0~ unique_word_n -1 ) 사이에서 n_samples 개의 정수 추출
        return_idx = np.random.choice(np.arange(self.unique_words_n), size=n_samples, replace=False, p=self.word_prob)

        # 각 idx 에 해당하는 단어를 찾아 list 형태로 반환
        return self.word_idx[return_idx].tolist()



