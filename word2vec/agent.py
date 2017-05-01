from data_proc import dataprocessing
import numpy as np
import pandas as pd


class agent:
    def __init__(self, learning_rate, n_window, vec_dim):
        # define hyperparameter
        self.n_window = n_window
        self.whole_window = self.n_window * 2 + 1
        self.learning_rate = learning_rate  # learning rate
        self.vec_dim = vec_dim  # 사용할 단어 벡터의 차원
        self.n_words = None # unique 단어 개수
        self.train = 10  # 반복학습횟수
        self.W = None  # 부모 weight 함수
        self.texts = None

        self.dataproc = dataprocessing(self.vec_dim)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        self.sigmoid = sigmoid

    def initialize(self):

        # 말뭉치에서 array type으로 단어 뭉치 가져오기
        self.texts = self.dataproc.read_corpus()

        # 웨이트 들고오기
        self.W = self.dataproc.build_word_matrix(texts, self.vec_dim)

        # Corpus 에서 전체 unique 단어 개수 class variable에 저장
        self.n_words = self.W.shape[0]


    # train 작업 완료, jupyer notebook 통해서 실제 계산 가능 여부도 확인
    # input_word, positive_sample, negative_sample : list of word index
    # learning_rate : scalar
    def train(self, input_word, positive_sample, negative_sample, learning_rate):
        # define temporal input word vector and Weight 2
        W_2_idx = set(positive_sample + negative_sample)
        pos_idx = set(positive_sample)
        W_2 = self.W.ix[W_2_idx]
        input_word_vector = self.W.ix[input_word].values

        hidden_layer = input_word_vector

        # 원래 Transpose 하는 것이 맞으나 기존에 들고온 matrix가 이미 transpose 된 상태
        output_layer = self.sigmoid(np.dot(W_2, hidden_layer))  # 해당 단어가 positive sample과 함께 등장할 확률을 리턴한다

        # output layer 행 개수 정의
        output_size = len(W_2_idx)

        # define t
        t = pd.DataFrame(data = np.zeros(output_size), index = W_2_idx)
        t.ix[pos_idx] = 1
        t = t.values[:,0]

        # calculate first cost
        loss1 = self.sigmoid(output_layer - t)
        loss1 = loss1.reshape([output_size,1]) # dot 계산을 위해서 reshape
        hidden_layer = hidden_layer.reshape([1,self.vec_dim])
        E = np.dot(loss1, hidden_layer)

        W_2_updated = W_2 - (learning_rate * E )

        # reduced sum 구현 되도록 축 정하기
        EH = np.sum(np.dot(loss1.T, hidden_layer), axis = 0)

        # input word vector update
        input_word_vector = input_word_vector - learning_rate * EH.T

        self.W.ix[input_word] = input_word_vector
        self.W.ix[W_2_idx] = W_2_updated






