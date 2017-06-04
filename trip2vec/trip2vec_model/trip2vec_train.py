#-*- coding: utf-8 -*-
import re
from trip2vec import trip2vec

### Set Hyperparameter and settings ###
WINDOW = 10
MIN_COUNT = 5
PARALELL_SIZE = 8
LEARNING_RATE = 0.025
MIN_LEARNING_RATE = 0.025
ITERATIONS = 3
LOAD_MODEL = True
MODEL_NAME = 'TripAdvisor_trip2vec'
NEG_SAMPLES = 64
EMBEDDING_SIZE = 200
BATCH_SIZE = 50
OPTIMIZER = 'Adagrad' # 'Adagrad', 'SGD'
LOSS_TYPE = 'sampled_softmax_loss'  # 'sampled_softmax_loss'. 'nce_loss
CONCAT = True # 워드 아이디 트립 벡터를 합쳐서 쓸지 여부
CITY = 'paris'
PATH = './AdvModel/' + CITY





if __name__ == "__main__":
    # 클래스 불러오기
    trip = trip2vec(WINDOW, PARALELL_SIZE, LEARNING_RATE,
                 ITERATIONS, MODEL_NAME, LOAD_MODEL,
                    EMBEDDING_SIZE, NEG_SAMPLES, BATCH_SIZE,
                    OPTIMIZER, LOSS_TYPE, CONCAT, CITY)
    # CSV 불러오기
    data = trip.load_review()

    # 리뷰 읽어와 데이터 준비 및 여행지,리뷰어아이디, 워드 index 구해 np.array로 메모리 저장
    data, trip_ids, reviewer_ids, reviews = trip.label_dataframe(data)
    print("Read all review data ... .")

    # 단어, 아이디, 여행지 단어 뭉치들을 숫자 인덱스로 바꿔주기
    data_idx = trip.word2index(data, trip_ids, reviewer_ids, reviews)

    # 한 도시 학습 개시
    trip.initialize()
    if LOAD_MODEL :
        trip.restore(PATH)
        print("Tensorflow model is loaded")
    trip.fit(data_idx = data_idx)

    # save tensorflow model, trip/id/word vectors and each dictionaries.
    trip.save(PATH)

