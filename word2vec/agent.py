from data_proc import dataprocessing
import numpy as np


class agent:
    def __init__(self, n_sample, learning_rate, n_window, vec_dim):
        # define hyperparameter
        self.n_sample = n_sample
        self.n_window = n_window
        self.learning_rate = learning_rate  # learning rate
        self.vec_dim = vec_dim  # 사용할 단어 벡터의 차원
        self.n_words = None
        self.train = 10  # 반복학습횟수
        self.W = None  # 부모 weight 함수

        self.dataproc = dataprocessing()

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        self.sigmoid = sigmoid

    def text_surfing(self):
        whole_window = self.n_window * 2 + 1

        # 말뭉치에서 array type으로 단어 뭉치 가져오기
        texts = self.dataproc.read_corpus()

        # 웨이트 1, 2 들고오기
        self.W = self.dataproc.build_word_matrix(texts, self.vec_dim)

        self.n_words = self.W.shape[0]

        # 랜덤 샘플링할 index 가져오기
        negative_sample_idx = self.dataproc.negative_sampling(n_samples=self.n_sample)






    # input_word, positive_sample, negative_sample : list of word index
    # learning_rate : scalar
    def train(self, input_word, positive_sample, negative_sample, learning_rate):
        # define temporal input word vector and Weight 2
        W_2_idx = set(positive_sample + negative_sample)

        W_2 = self.W[W_2_idx]  # TODO np.array로 전환
        input_word_vector = self.W[input_word]  # TODO np.array로 전환

        hidden_layer = input_word_vector

        output_layer = self.sigmoid(np.dot(W_2.T, hidden_layer))  # 해당 단어가 positive sample과 함께 등장할 확률을 리턴한다

        # output layer 행 개수 정의
        output_size = len(W_2_idx)

        # define t
        t = None  # TODO t define 하기

        E = np.dot((output_layer - t))

        W_2_updated = W_2 - (learning_rate * E * hidden_layer)

        # reduced sum 구현 되도록 축 정하기
        EH = np.sum(np.dot(E, W_2.T))

        # input word vector update
        input_word_vector = input_word_vector - learning_rate * EH.T

        self.W[input_word] = input_word_vector
        self.W[W_2_idx] = W_2_updated

        return

