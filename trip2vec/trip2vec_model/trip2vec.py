import tensorflow
import nltk
import pandas as pd
from sklearn import utils
import re


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
        punctuation = ".,?!:;(){}[]"
        for char in punctuation: corpus = corpus.replace(char, "")
        corpus = corpus.lower()
        corpus = corpus.split()
        return corpus

    # Attraction preprocessing
    def treat_attraction(self, s):
        m = ATTRACTION_RE.search(s)
        if m: s = re.sub(m.group(), '', s)
        return s

    # 리뷰 파일 csv 를 불러오는 함수
    def load_review(self):
        raw_data = pd.read_csv('./trip.csv', header=0)
        return raw_data

    # word, reviewer id, trip site 데이터 학습할 수 있도록 만들어 주는 함수
    def build_data_set(self):

        pass


class trip2vec(preprocess):
    def __init__(self):
        self.window = WINDOW
        self.paralell_size = PARALELL_SIZE
        self.learning_rate = LEARNING_RATE
        self.iterations = ITERATIONS
        self.model_name = MODEL_NAME
        self.load_model = LOAD_MODEL
        pass
    

    
    # word, reviewer, trip site 매트릭스를 생성하는 함수
    # 어느 조 발표에서 처럼 값을 들고오지말고 그대로 사용하는 방법 사용해보기
    def build_matrix(self):
        
        pass
    
    # tensorflow 그래프를 빌딩하는 함수
    def build_tensor_graph(self):
        
        pass