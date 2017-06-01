
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import nltk
import pandas as pd
import numpy as np
import re
import os
import json
import collections
from itertools import compress
import math


# 리뷰 전처리 클래스 (구두점 제거, 텍스트 보정, 리뷰 불러오기, 데이터 준비하기)
class preprocess():
    def __init__(self, WINDOW, PARALELL_SIZE, LEARNING_RATE,
                 ITERATIONS, MODEL_NAME, LOAD_MODEL, VECTOR_SIZE,
                 EMBEDDING_SIZE, NEG_SAMPLES, BATCH_SIZE,
                 OPTIMIZER, LOSS_TYPE, CONCAT, CITY):
        self.window_size = WINDOW
        self.paralell_size = PARALELL_SIZE
        self.learning_rate = LEARNING_RATE
        self.iterations = ITERATIONS
        self.model_name = MODEL_NAME
        self.load_model = LOAD_MODEL
        self.vector_size = VECTOR_SIZE
        self.ATTRACTION_RE = re.compile('.*about.')
        self.embedding_size_w = EMBEDDING_SIZE
        self.embedding_size_i = EMBEDDING_SIZE
        self.embedding_size_t = EMBEDDING_SIZE
        self.batch_size = BATCH_SIZE
        self.optimize = OPTIMIZER
        self.loss_type = LOSS_TYPE
        self.n_neg_samples = NEG_SAMPLES
        self.concat = CONCAT
        self.city = CITY



    # Preprocessing function
    def cleanText(self, corpus):
        punctuation = ".,?!:;(){}[]\"'"
        for char in punctuation: corpus = corpus.replace(char, "")
        corpus = corpus.lower()
        corpus = corpus.split()
        return corpus

    # Attraction preprocessing
    def treat_attraction(self, s):
        m = self.ATTRACTION_RE.search(s)
        if m: s = re.sub(m.group(), '', s)
        return s

    # 여행지 아이디, 리뷰어 아이디, 리뷰 말뭉치를 하나의 리스트에 저장
    def label_dataframe(self, df):
        ret = []
        trip_ids = set(); reviewer_ids = set(); reviews = set()
        for idx, row in df.iterrows():
            temp_corpus = self.cleanText(row['review_text'])
            trip_id = row['attraction']
            reviewer_id = row['reviewer_id']
            one_review = [trip_id,reviewer_id,temp_corpus]
            ret.append(one_review)

            # Build trip site, reviewer id, review index
            trip_ids = trip_ids.union([trip_id])
            reviewer_ids = reviewer_ids.union([reviewer_id])
            reviews = reviews.union(temp_corpus)

        # 각 분야별 유니크한 단어들 모아 반환하기
        trip_ids = np.array(list(trip_ids))
        reviewer_ids = np.array(list(reviewer_ids))
        reviews = np.array(list(reviews))

        # 각 섹터별별 유니크한 단 길이 클래스 변수에 저장
        self.trip_size = len(trip_ids)
        self.id_size = len(reviewer_ids)
        self.vocabulary_size = len(reviews)

        return (ret, trip_ids, reviewer_ids, reviews)


    # 리뷰 파일 csv 를 불러오는 함수
    def load_review(self):
        raw_data = pd.read_csv('./trip.csv', header=0)
        return raw_data



class trip2vec(preprocess):
    def __init__(self):
        pass

    # word, reviewer, trip site 매트릭스를 생성하는 함수
    def build_matrix(self):
        self.trip_matrix = np.random.uniform(-1,1,[self.trip_len, self.vector_size])
        self.id_matrix = np.random.uniform(-1,1,[self.id_len, self.vector_size])
        self.word_matrix = np.random.uniform(-1,1,[self.review_len, self.vector_size])

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
        batch = np.ndarray(shape=(batch_size, window_size + 1), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
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


    def _init_graph(self):
        '''
        Init a tensorflow Graph containing:
        input data, variables, model, loss function, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size, self.window_size + 1])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
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
            if self.loss_type == 'sampled_softmax_loss':
                loss = tf.nn.sampled_softmax_loss(self.weights, self.biases, self.embed,
                                                  self.train_labels, self.n_neg_samples, self.vocabulary_size)
            elif self.loss_type == 'nce_loss':
                loss = tf.nn.nce_loss(self.weights, self.biases, self.embed,
                                      self.train_labels, self.n_neg_samples, self.vocabulary_size)
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
            self.normalized_doc_embeddings = self.id_embeddings / norm_i

            norm_t = tf.sqrt(tf.reduce_sum(tf.square(self.trip_embeddings), 1, keep_dims=True))
            self.normalized_doc_embeddings = self.trip_embeddings / norm_t

            # init op
            self.init_op = tf.global_variables_initializer()
            # create a saver
            self.saver = tf.train.Saver()

    def fit(self, docs):
        '''
        words: a list of words.
        '''
        # pre-process words to generate indices and dictionaries
        doc_ids, word_ids = self._build_dictionaries(docs)

        # with self.sess as session:
        session = self.sess

        session.run(self.init_op)

        average_loss = 0
        print("Initialized")
        for step in range(self.n_steps):
            batch_data, batch_labels = self.generate_batch(doc_ids, word_ids,
                                                           self.batch_size, self.window_size)
            feed_dict = {self.train_dataset: batch_data, self.train_labels: batch_labels}
            op, l = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
            average_loss += l
            if step % 2000 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f' % (step, average_loss))
                average_loss = 0

        # bind embedding matrices to self
        self.word_embeddings = session.run(self.normalized_word_embeddings)
        self.doc_embeddings = session.run(self.normalized_doc_embeddings)

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
        estimator = Doc2Vec(**params)
        estimator._restore(path)
        # evaluate the Variable embeddings and bind to estimator
        estimator.word_embeddings = estimator.sess.run(estimator.normalized_word_embeddings)
        estimator.doc_embeddings = estimator.sess.run(estimator.normalized_doc_embeddings)
        # bind dictionaries
        estimator.dictionary = json.load(open(os.path.join(path_dir, 'model_dict.json'), 'rb'))
        reverse_dictionary = json.load(open(os.path.join(path_dir, 'model_rdict.json'), 'rb'))
        # convert indices loaded from json back to int since json does not allow int as keys
        estimator.reverse_dictionary = {int(key): val for key, val in reverse_dictionary.items()}

        return estimator

    # tensorflow 그래프를 빌딩하는 함수
    def build_tensor_graph(self):
        
        pass




if __name__ == "__main__":
    # csv 불러오기
    trip = trip2vec()
    data = trip.load_review()

    # Paris data 만 가져오기
    data = data[data['city'] == 'Paris']

    # Attraction 앞 about 제거
    data['attraction'] = data['attraction'].apply(trip.treat_attraction)

    # 리뷰 읽어와 데이터 준비 및 여행지,리뷰어아이디, 워드 index 구해 np.array로 메모리 저장
    data, trip_ids, reviewer_ids, reviews = trip.label_dataframe(data)
    print("Read all review data ... start build matrix ...")

    #TODO 필요없을지도 ... 다시 ref 잘 살펴보기기
   # np.array로 각 여행지, 리뷰어아이디, 단어 벡터 생성
    trip.build_matrix()

    # tensorflow graph 생성
    ##TODO 레퍼런스 자료 잘 갖춰져 있어 짜맞춰 넣어보기 화이팅~!!