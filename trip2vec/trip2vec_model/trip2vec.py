#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import gensim
import pandas as pd
import numpy as np
import re
import os
import json
import collections
from itertools import compress
import math
import time
from gensim.parsing.preprocessing import PorterStemmer


# 리뷰 전처리 클래스 (구두점 제거, 텍스트 보정, 리뷰 불러오기, 데이터 준비하기)
class preprocess():
    def __init__(self) :
        self.porter = PorterStemmer()


    # Preprocessing function
    def cleanText(self, corpus):
        punctuation = "&.,?!:;(){}[–-]\"'`0123456789"
        for char in punctuation: corpus = corpus.replace(char, "")
        corpus = corpus.lower()
        corpus = corpus.split()
        ret = []
        for word in corpus :
            ret.append(self.porter.stem(word))
        return ret

    def cleanText2(self, corpus):
        punctuation = "&\n.,?!:;(){}[–-]\"'` "
        for char in punctuation: corpus = corpus.replace(char, "")
        return corpus

    def isnumalpha(self, string):
        num_alpha = re.compile('^[a-zA-Z0-9]+$')
        return bool(num_alpha.match(string))

    # 여행지 아이디, 리뷰어 아이디, 리뷰 말뭉치를 하나의 리스트에 저장
    def label_dataframe(self, df):
        st = time.time()
        print('Start labeling')
        ret = []
        trip_ids = []; reviewer_ids = []; reviews = []

        for idx, row in df.iterrows():
            temp_corpus = self.cleanText(row['review_text'])
            trip_id = row['attraction']
            reviewer_id = row['user_id']
            one_review = [trip_id,reviewer_id,temp_corpus]
            ret.append(one_review)

            # Build trip site, reviewer id, review index
            trip_ids.append(trip_id)
            reviewer_ids.append(reviewer_id)
            reviews.extend(temp_corpus)

        # 각 분야별 유니크한 단어들 모아 반환하기
        trip_ids = np.array(list(set(trip_ids)))
        reviewer_ids = np.array(list(set(reviewer_ids)))
        reviews = np.array(list(set(reviews)))

        print("It took %.2f seconds to labeling data"%(time.time()-st))
        return (ret, trip_ids, reviewer_ids, reviews)


    # 리뷰 파일 csv 를 불러오는 함수
    def load_review(self):
        # csv 불러와 pandas dataframe에 저장
        path = './' + self.city + '.csv'
        raw_data = pd.read_csv(path, header=0)

        # 영문이 아닌 리뷰를 걸러주기
        data_for_idx = raw_data['review_text'].apply(self.cleanText2)
        bool_idx = data_for_idx.apply(self.isnumalpha)

        raw_data = raw_data[bool_idx]

        return raw_data


class trip2vec(preprocess):

    def __init__(self, WINDOW, PARALELL_SIZE, LEARNING_RATE,
                 ITERATIONS, MODEL_NAME, LOAD_MODEL, VECTOR_SIZE,
                 EMBEDDING_SIZE, NEG_SAMPLES, BATCH_SIZE,
                 OPTIMIZER, LOSS_TYPE, CONCAT, CITY):
        self.porter = PorterStemmer()
        self.window_size = WINDOW
        self.paralell_size = PARALELL_SIZE
        self.learning_rate = LEARNING_RATE
        self.iterations = ITERATIONS
        self.model_name = MODEL_NAME
        self.load_model = LOAD_MODEL
        self.vector_size = VECTOR_SIZE
        self.embedding_size_w = EMBEDDING_SIZE
        self.embedding_size_i = EMBEDDING_SIZE
        self.embedding_size_t = EMBEDDING_SIZE
        self.batch_size = BATCH_SIZE
        self.optimize = OPTIMIZER
        self.loss_type = LOSS_TYPE
        self.n_neg_samples = NEG_SAMPLES
        self.concat = CONCAT
        self.city = CITY
        self.Dict = gensim.corpora.dictionary.Dictionary
        self.vocabulary_size = None
        self.id_size = None
        self.vocabulary_size = None

        pass


    def generate_batch_pvdm(self, word_ids, id_ids, trip_ids, batch_size, window_size):
        '''
        Batch generator for PV-DM (Distributed Memory Model of Paragraph Vectors).
        batch should be a shape of (batch_size, window_size+1)

        Parameters
        ----------
        word_ids: list of word indices
        id_ids : list of reviewer id indices
        trip_ids: list of trip site indices
        batch_size: number of words in each mini-batch
        window_size: number of leading words before the target word
        '''
        global data_index
        assert batch_size % window_size == 0
        batch = np.ndarray(shape=(batch_size, window_size + 1), dtype=np.float32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.float32)
        span = window_size + 1
        buffer = collections.deque(maxlen=span)  # used for collecting word_ids[data_index] in the sliding window
        buffer_id = collections.deque(maxlen=span)
        buffer_trip = collections.deque(maxlen=span)  # collecting id of documents in the sliding window
        # collect the first window of words
        for _ in range(span):
            buffer.append(word_ids[data_index])
            buffer_id.append(trip_ids[data_index])
            buffer_trip.append(trip_ids[data_index])
            data_index = (data_index + 1) % len(word_ids)

        mask = [1] * span
        mask[-1] = 0
        i = 0
        while i < batch_size:
            if len(set(buffer_trip)) == 1 and len(set(buffer_id)) == 1:
                id_id = buffer_id[-1]
                trip_id = buffer_trip[-1]
                # all leading words and the doc_id
                batch[i, :] = list(compress(buffer, mask)) + +[id_id] + [trip_id]
                labels[i, 0] = buffer[-1]  # the last word at end of the sliding window
                i += 1
            # move the sliding window
            buffer.append(word_ids[data_index])
            buffer_id.append(id_ids[data_index])
            buffer_trip.append(trip_ids[data_index])
            data_index = (data_index + 1) % len(word_ids)

        return batch, labels

    # set graph initialized
    def initialize(self):
        # init all variables in a tensorflow graph
        self._init_graph()
        # create a session
        self.sess = tf.Session(graph=self.graph)

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing:
        input data, variables, model, loss function, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size, self.window_size + 1])
            self.train_labels = tf.placeholder(tf.float32, shape=[self.batch_size, 1])
            # Variables.
            # embeddings for words, W in paper
            self.word_embeddings = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_size_w], -1.0, 1.0))

            # embedding for reviewer id
            self.id_embeddings = tf.Variable(
                tf.random_uniform([self.id_size, self.embedding_size_i], -1.0, 1.0))

            # embedding for trip sites
            self.trip_embeddings = tf.Variable(
                tf.random_uniform([self.trip_size, self.embedding_size_t], -1.0, 1.0))

            if self.concat:  # concatenating word vectors and doc vector
                combined_embed_vector_length = self.embedding_size_w * self.window_size + self.embedding_size_i + self.embedding_size_t
            else:  # concatenating the average of word vectors and the doc vector
                combined_embed_vector_length = self.embedding_size_w + self.embedding_size_i + self.embedding_size_t

            # softmax weights, W and D vectors should be concatenated before applying softmax
            self.weights = tf.Variable(
                tf.truncated_normal([self.vocabulary_size, combined_embed_vector_length],
                                    stddev=1.0 / math.sqrt(combined_embed_vector_length)))
            # softmax biases
            self.biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            # Model.
            # Look up embeddings for inputs.
            # shape: (batch_size, embeddings_size)
            embed = []  # collect embedding matrices with shape=(batch_size, embedding_size)
            if self.concat:
                for j in range(self.window_size):
                    embed_w = tf.nn.embedding_lookup(self.word_embeddings, self.train_dataset[:, j])
                    embed.append(embed_w)
            else:
                # averaging word vectors
                embed_w = tf.zeros([self.batch_size, self.embedding_size_w])
                for j in range(self.window_size):
                    embed_w += tf.nn.embedding_lookup(self.word_embeddings, self.train_dataset[:, j])
                embed.append(embed_w)

            embed_i = tf.nn.embedding_lookup(self.id_embeddings, self.train_dataset[:, self.window_size])
            embed_t = tf.nn.embedding_lookup(self.trip_embeddings, self.train_dataset[:, self.window_size])
            embed.append(embed_i)
            embed.append(embed_t)
            # concat word and doc vectors
            self.embed = tf.concat(values = embed, axis = 1)

            # Compute the loss, using a sample of the negative labels each time.
            print(self.embed.shape, self.train_labels.shape, self.train_dataset.shape)
            if self.loss_type == 'sampled_softmax_loss':
                loss = tf.nn.sampled_softmax_loss(weights = self.weights,
                                                  biases = self.biases,
                                                  inputs = self.embed,
                                                  labels = self.train_labels,
                                                  num_sampled = self.n_neg_samples,
                                                  num_classes = self.vocabulary_size)
            elif self.loss_type == 'nce_loss':
                loss = tf.nn.nce_loss(weights = self.weights,
                                                  biases = self.biases,
                                                  inputs = self.embed,
                                                  labels = self.train_labels,
                                                  num_sampled = self.n_neg_samples,
                                                  num_classes = self.vocabulary_size)
            self.loss = tf.reduce_mean(loss)

            # Optimizer.
            if self.optimize == 'Adagrad':
                self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)
            elif self.optimize == 'SGD':
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

            # Compute the similarity between minibatch examples and all embeddings.
            # We use the cosine distance:
            norm_w = tf.sqrt(tf.reduce_sum(tf.square(self.word_embeddings), 1, keep_dims=True))
            self.normalized_word_embeddings = self.word_embeddings / norm_w

            norm_i = tf.sqrt(tf.reduce_sum(tf.square(self.id_embeddings), 1, keep_dims=True))
            self.normalized_id_embeddings = self.id_embeddings / norm_i

            norm_t = tf.sqrt(tf.reduce_sum(tf.square(self.trip_embeddings), 1, keep_dims=True))
            self.normalized_trip_embeddings = self.trip_embeddings / norm_t

            # init op
            self.init_op = tf.global_variables_initializer()
            # create a saver
            self.saver = tf.train.Saver()

    # 각 유니크 단어사전을 통해 단어로 구성된 문서들을 모두 index로 바꿔주기
    def word2index(self, data, trip_ids, reviewer_ids, reviews):
        st = time.time()
        data_idx = []

        # 각 섹터별별 유니크한 단어 길이 클래스 변수에 저장
        self.trip_size = len(trip_ids)
        self.id_size = len(reviewer_ids)
        self.vocabulary_size = len(reviews)
        print("Unique trip sites : {}, reviewer ids : {}, words : {}"\
              .format(self.trip_size, self.id_size, self.vocabulary_size))

        # Build trip, id, word dictionary
        self.trip_dict = self.Dict([trip_ids]).token2id
        self.id_dict = self.Dict([reviewer_ids]).token2id
        self.word_dict = self.Dict([reviews]).token2id
        for review in data :
            trip_id_idx = self.trip_dict[review[0]]
            try :
                reviewer_id_idx = self.id_dict[review[1]]
            except : continue

            review_idx = []
            for word in review[2] :
                review_idx.append(self.word_dict[word])

            review_len = len(review_idx)
            trip_id_idx = [trip_id_idx] * review_len
            reviewer_id_idx = [reviewer_id_idx] * review_len

            data_idx.append([trip_id_idx, reviewer_id_idx, review_idx])
        print("It took %.2f seconds to make index"%(time.time() - st))
        return data_idx

    def fit(self, data_idx):
        '''
        trip_ids : a list of same trip ids in one review
        id_ids : a list of same ids in one review
        word_ids: a list of words in one review.
        '''
        # with self.sess as session:
        session = self.sess
        session.run(self.init_op)
        print("Initialized")

        # k번 반복 학습
        for k in self.iterations :
            # set for index and calculate loss
            average_loss = 0; i = 0; total_step = len(data_idx)
            for data in data_idx :
                trip_ids = data[0]
                id_ids = data[1]
                word_ids = data[2]

                batch_data, batch_labels = self.generate_batch_pvdm(trip_ids= trip_ids,
                                                                    id_ids = id_ids,
                                                                    word_ids = word_ids,
                                                                    batch_size = self.batch_size,
                                                                    window_size = self.window_size)
                feed_dict = {self.train_dataset: batch_data, self.train_labels: batch_labels}
                op, l = session.run([self.optimizer, self.loss], feed_dict=feed_dict)

                average_loss += l
                if i % 5000 == 0:
                    if i > 0:
                        average_loss = average_loss / 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Learning %.3f %% Average loss at step %d: %f' % (i/total_step,i, average_loss))
                    average_loss = 0

        # bind embedding matrices to self
        self.word_embeddings = session.run(self.normalized_word_embeddings)
        self.id_embeddings = session.run(self.normalized_id_embeddings)
        self.trip_embeddings = session.run(self.normalized_trip_embeddings)

        return self

    def save(self, path):
        '''
        To save trained model and its params.
        '''
        save_path = self.saver.save(self.sess,
                                    os.path.join(path, 'model.ckpt'))
        print("Model saved in file: %s" % save_path)
        return save_path

    def _restore(self, path):
        with self.graph.as_default():
            self.saver.restore(self.sess, path)

    @classmethod
    def restore(cls, path):
        '''
        To restore a saved model.
        '''
        # load params of the model
        path_dir = os.path.dirname(path)
        params = json.load(open(os.path.join(path_dir, 'model_params.json'), 'rb'))
        # init an instance of this class
        estimator = trip2vec(**params)
        estimator._restore(path)
        return estimator
