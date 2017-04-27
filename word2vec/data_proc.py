import nltk
import numpy as np
import pandas as pd

class dataprocessing :
    def __init__(self, vector_dim):
        self.corpus_add = './sample.txt'
        self.word_prob = None  #단어 확률 분포 변수( pandas dataframe 할당 예정 )
        pass


    def read_corpus(self):
        # get corpus
        corpus_file = open(self.corpus_add)
        text = corpus_file.readlines()
        # tokenize the corpus
        tokens = nltk.word_tokenize(text)
        return tokens

    def build_word_matrix(self, tokens, vector_dim):

        # build standard word matrix
        # initialize the elements values
        word_matrix = pd.DataFrame(data = np.random.normal(0,0.5,size = (len(set(tokens),vector_dim))),
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
        self.word_prob = word_temp1 / denom   # 클래스 변수에 바로 할당

        # weight matrix 2개 관리하기 때문에 한번 연산으로 두 개 반환
        return word_matrix

    # 총단어의 갯수(idx), sampling할 단어 수 n개 입력해 n개의 idx를 받아온다
    #TODO negative samping  word_prob 받아와서 랜덤으로 단어 뽑아서 리스트 반환하는 작업 완료
    def negative_sampling(self, n_unique_words, n_samples):
        return np.random.randint(low = 0, high = n_unique_words, size = n_samples)



