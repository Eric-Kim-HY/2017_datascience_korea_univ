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
    def train(self, positive_sample, negative_sample, label_vector, learning_rate):

        def relu(x):
            output = max(0,x)
            return output

        def softmax(x):
            e_x = np.exp(x)
            return e_x / e_x.sum()

        # input data (@layer_0)
        input_sample = positive_sample + negative_sample

        # output data (layer_2 값과 비교할 원래 값, one-hot 형식)
        input_sample_onehot = label_vector

        # forward propagation
        layer_0 = input_sample                              # (dim: 1xV)인 input sample을 넣는다
        positive_sample_vector = layer_0.T.dot(agent.W_1) # positive sample의 벡터값을 리턴한다 (dim: 1xN)
        layer_1 = positive_sample_vector
        layer_2 = softmax(relu(np.dot(layer_1, agent.W_2))) # 해당 단어가 positive sample과 함께 등장할 확률을 리턴한다

        # update weight matrix
        layer_2_error = layer_2 - input_sample_onehot
        W_2 = W_2 - learning_rate * (layer_1.T.dot(layer_2_error))
        W_1 = W_1 - learning_rate *

        return W_1

