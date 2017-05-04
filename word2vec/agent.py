from data_proc import dataprocessing
import numpy as np
import pandas as pd
import tensorflow as tf


class agent:
    def __init__(self, learning_rate, n_window, vec_dim):
        # define hyperparameter
        self.n_window = n_window
        self.whole_window = self.n_window * 2 + 1
        self.learning_rate = learning_rate  # learning rate
        self.vec_dim = vec_dim  # 사용할 단어 벡터의 차원
        self.n_words = None # unique 단어 개수
        self.W = None  # 부모 weight 함수
        self.texts = None
        self.n_total_words = None  #중복 포함한 전체 단어 갯수
        self.dataproc = dataprocessing()
        self.unique_words_n = None # 총 단어 종류 수
        self.word_idx = None # 단어 확률분포에 대응되는 idx word
        self.word_prob = None  # 단어 확률 분포 변수( list 할당 예정 )
        self.sess = tf.Session() # tensorflow graph 인자 (train_tf에서 사용)

        def sigmoid(x):
            return 1 / (1 + np.round(np.exp(-x),10))

        self.sigmoid = sigmoid

    def initialize(self):

        print("Starts initialize")

        # 말뭉치에서 array type으로 단어 뭉치 가져오기
        self.texts = self.dataproc.read_corpus()
        print("Loaded the whole corpus")

        self.n_total_words = len(self.texts)

        # 웨이트 들고오기
        self.W, self.unique_words_n, self.word_idx, self.word_prob =\
                self.dataproc.build_word_matrix(self.texts, self.vec_dim)
        print("Built the parent matrix")

        # Corpus 에서 전체 unique 단어 개수 class variable에 저장
        self.n_words = self.W.shape[0]


        print("Initialized!")


    # sampling할 단어 수 n개 입력해 n개의 idx words 를 받아온다
    def negative_sampling(self, n_samples):

        # word_prob 확률을 이용해서 (0~ unique_word_n -1 ) 사이에서 n_samples 개의 정수 추출
        return_idx = np.random.choice(np.arange(self.unique_words_n), size=n_samples, replace=False, p=self.word_prob)

        # 각 idx 에 해당하는 단어를 찾아 list 형태로 반환
        return self.word_idx[return_idx].tolist()

    # train 작업 완료, jupyter notebook 통해서 실제 계산 가능 여부도 확인
    # input_word, positive_sample, negative_sample : list of word index
    # learning_rate : scalar
    def train(self, input_word, positive_sample, negative_sample):
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

        W_2_updated = W_2 - (self.learning_rate * E )

        # reduced sum 구현 되도록 축 정하기
        EH = np.sum(np.dot(loss1, hidden_layer), axis = 0)

        # input word vector update
        input_word_vector = input_word_vector - self.learning_rate * EH.T

        self.W.ix[input_word] = input_word_vector
        self.W.ix[W_2_idx] = W_2_updated

    # Tensorflow graph 구축
    def build_network(self):
        ### hidden layer, W_2, t 담을 place holder 정의
        # Shape [ vec_dim, 1 ]
        self.hidden_layer = tf.placeholder(tf.float32, [self.vec_dim, None]) # Size (vector dimension)

        # Shape [n_sample,vec_dim]
        self.W_2 = tf.placeholder(tf.float32,[None, self.vec_dim])

        # Shape [n_sample, 1]
        self.t = tf.placeholder(tf.float32, [None,1])


        ### Build Graph
        # Shape [n_sample, 1]
        self.output_layer = 1/(tf.exp(-(tf.matmul(self.W_2, self.hidden_layer))) + 1)  # 해당 단어가 positive sample과 함께 등장할 확률을 리턴한다

        # Shape [n_sample, 1] calculate first cost
        self.loss1 = 1/(tf.exp(-(self.output_layer - self.t)) + 1)

        # Shape [n_sample, vec_dim]
        self.E = tf.matmul(self.loss1, self.hidden_layer, transpose_b=True)

        # Shape [n_sample, vec_dim]
        self.W_2_updated = self.W_2 - (self.learning_rate * self.E )

        # Shape [100], reduced sum 구현 되도록 축 정하기
        self.EH = tf.reduce_sum(self.E, axis = 0)

        # Shape [100], input word vector update
        self.input_word_updated = tf.transpose(self.hidden_layer) - self.learning_rate * self.EH

    # train with tensorflow graph
    def train_tf(self, input_word, positive_sample, negative_sample):
        # define temporal input word vector and Weight 2
        W_2_idx = set(positive_sample + negative_sample)
        pos_idx = set(positive_sample)
        W_2 = self.W.ix[W_2_idx]
        input_word_vector = self.W.ix[input_word].values.reshape([self.vec_dim,1])

        hidden_layer = input_word_vector

        # output layer 행 개수 정의
        output_size = len(W_2_idx)

        # define t
        t = pd.DataFrame(data = np.zeros(output_size), index = W_2_idx)
        t.ix[pos_idx] = 1
        t = t.values[:,0].reshape([t.shape[0],1])

        # update 된 vector를 tensorflow graph 실행 시키며 바로 할당
        self.W.ix[input_word], self.W.ix[W_2_idx] = \
        self.sess.run([self.input_word_updated,self.W_2_updated],
                      feed_dict= {self.hidden_layer : hidden_layer,
                                  self.W_2 : W_2,
                                  self.t : t})


    def save_model(self):
        self.W.to_csv('./wordvector.csv')


    def load_model(self):
        return pd.read_csv('./wordvector.csv', header= 0, index_col = 0)




