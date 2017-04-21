from data_proc import dataprocessing
import numpy as np


class agent :
    def __init__(self, n_sample, learning_rate, n_window, vec_dim):
        # define hyperparameter
        self.n_sample = n_sample
        self.n_window = n_window
        self.learning_rate = learning_rate
        self.vec_dim = vec_dim
        self.n_words = None


        self.dataproc = dataprocessing()



    def text_surfing(self):
        whole_window = self.n_window*2 + 1

        # 말뭉치에서 array type으로 단어 뭉치 가져오기
        texts = self.dataproc.read_corpus()

        # 웨이트 1, 2 들고오기
        self.W_1, self.W_2 = self.dataproc.build_word_matrix(texts, self.vec_dim)

        self.n_words = self.W_1.shape[0]

        # 랜덤 샘플링할 index 가져오기
        weigh_idx = self.dataproc.negative_sampling(n_unique_words= self.n_words, n_samples = self.n_sample)