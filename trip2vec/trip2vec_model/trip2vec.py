import tensorflow
import nltk
import pandas as pd
import numpy as np
from sklearn import utils
import re
import gensim


### Set Hyperparameter and settings ###
VECTOR_SIZE = 300
WINDOW = 10
MIN_COUNT = 5
PARALELL_SIZE = 8
LEARNING_RATE = 0.025
MIN_LEARNING_RATE = 0.025
ITERATIONS = 20
LOAD_MODEL = False
ATTRACTION_RE = re.compile('.*about.')
MODEL_NAME = 'TripAdvisor_trip2vec'

# 리뷰 전처리 클래스 (구두점 제거, 텍스트 보정, 리뷰 불러오기, 데이터 준비하기)
class preprocess():
    def __init__(self):
        pass

    # Preprocessing function
    def cleanText(self, corpus):
        punctuation = ".,?!:;(){}[]\"'"
        for char in punctuation: corpus = corpus.replace(char, "")
        corpus = corpus.lower()
        corpus = corpus.split()
        return corpus

    # Attraction preprocessing
    def treat_attraction(self, s):
        m = ATTRACTION_RE.search(s)
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
        self.trip_len = len(trip_ids)
        self.id_len = len(reviewer_ids)
        self.review_len = len(reviews)

        return (ret, trip_ids, reviewer_ids, reviews)


    # 리뷰 파일 csv 를 불러오는 함수
    def load_review(self):
        raw_data = pd.read_csv('./trip.csv', header=0)
        return raw_data



class trip2vec(preprocess):
    def __init__(self):
        self.window = WINDOW
        self.paralell_size = PARALELL_SIZE
        self.learning_rate = LEARNING_RATE
        self.iterations = ITERATIONS
        self.model_name = MODEL_NAME
        self.load_model = LOAD_MODEL
        self.vector_size = VECTOR_SIZE

    # word, reviewer, trip site 매트릭스를 생성하는 함수
    # 어느 조 발표에서 처럼 값을 들고오지말고 그대로 사용하는 방법 사용해보기
    def build_matrix(self):
        self.trip_matrix = np.random.uniform(-1,1,[self.trip_len, self.vector_size])
        self.id_matrix = np.random.uniform(-1,1,[self.id_len, self.vector_size])
        self.word_matrix = np.random.uniform(-1,1,[self.review_len, self.vector_size])

    
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