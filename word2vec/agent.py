from data_proc import dataprocessing
import numpy as np


class agent :
    def __init__(self, n_sample, learning_rate, n_window, vec_dim):
        # define hyperparameter
        self.n_sample = n_sample
        self.n_window = n_window
        self.learning_rate = learning_rate  #learning rate
        self.vec_dim = vec_dim # 사용할 단어 벡터의 차원
        self.n_words = None
        self.train = 10  # 반복학습횟수

        self.dataproc = dataprocessing()



    def text_surfing(self):
        whole_window = self.n_window*2 + 1

        # 말뭉치에서 array type으로 단어 뭉치 가져오기
        texts = self.dataproc.read_corpus()

        # 웨이트 1, 2 들고오기
        self.W_1 = self.dataproc.build_word_matrix(texts, self.vec_dim)

        self.n_words = self.W_1.shape[0]

        # 랜덤 샘플링할 index 가져오기
        weight_idx = self.dataproc.negative_sampling(n_samples = self.n_sample)


    # sampleword : output + negative sample words
    # inputword, sampleword, label vector : np.array, learning_rate : scalar
    def train(self, input_word, sample_word, label_vector, learning_rate):

        def relu(x):
            return max(0, x)

        # forward propagation
        layer_0 = input_word                  # 0번째 layer == input matrix
        layer_1 = word_vector  # 1번째 layer: layer0에 W_0 행렬곱
        layer_2 = relu(np.dot(layer_1, dataprocessing.build_word_matrix()))  # 2번째 layer: layer1에 W_1 행렬곱

        # calculate error
        layer_2_error = layer_2 - y  # error는 원래값(y)과의 차이

        layer_1_error = layer_2_error.dot(W_1.T)  # reLU의 미분값은 양의 값에서 항상 1 이므로 'delta = error'

        #       layer_1_delta = layer_1_error*sig2deriv(layer_1)

        # update weight matrix
        W_1 -= learning_rate * (layer_1.T.dot(layer_2_error))

        #
        return updated_sample_word

