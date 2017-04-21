from data_proc import dataprocessing
import numpy as np


class agent :
    def __init__(self, n_sample, learning_rate, n_window, vec_dim):
        # define hyperparameter
        self.n_sample = n_sample
        self.n_window = n_window
        self.learning_rate = learning_rate
        self.vec_dim = vec_dim



        self.dataproc = dataprocessing(self.vec_dim)



    def text_surfing(self):
        whole_window = self.n_window*2 + 1
        texts = np.array(self.dataproc.read_corpus())
        n_texts = len(texts)