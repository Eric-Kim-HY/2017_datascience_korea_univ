import re
from trip2vec_model.trip2vec import trip2vec

### Set Hyperparameter and settings ###
VECTOR_SIZE = 300
WINDOW = 10
MIN_COUNT = 5
PARALELL_SIZE = 8
LEARNING_RATE = 0.025
MIN_LEARNING_RATE = 0.025
ITERATIONS = 20
LOAD_MODEL = False
MODEL_NAME = 'TripAdvisor_trip2vec'
NEG_SAMPLES = 64
EMBEDDING_SIZE = 200
BATCH_SIZE = 50
OPTIMIZER = 'Adagrad' # 'Adagrad', 'SGD'
LOSS_TYPE = 'sampled_softmax_loss'  # 'sampled_softmax_loss'. 'nce_loss
CONCAT = True # 워드 아이디 트립 벡터를 합쳐서 쓸지 여부
CITY = 'paris'





if __name__ == "__main__":
    # csv 불러오기
    trip = trip2vec(WINDOW, PARALELL_SIZE, LEARNING_RATE,
                 ITERATIONS, MODEL_NAME, LOAD_MODEL, VECTOR_SIZE,
                    EMBEDDING_SIZE, NEG_SAMPLES, BATCH_SIZE,
                    OPTIMIZER, LOSS_TYPE, CONCAT, CITY)
    data = trip.load_review()

    # Paris data 만 가져오기
    data = data[data['city'] == 'Paris']

    # 리뷰 읽어와 데이터 준비 및 여행지,리뷰어아이디, 워드 index 구해 np.array로 메모리 저장
    data, trip_ids, reviewer_ids, reviews = trip.label_dataframe(data)
    print("Read all review data ... .")

    trip.build_batch()

    #TODO 필요없을지도 ... 다시 ref 잘 살펴보기기
   # np.array로 각 여행지, 리뷰어아이디, 단어 벡터 생성
    trip.build_matrix()

    # tensorflow graph 생성
    ##TODO 레퍼런스 자료 잘 갖춰져 있어 짜맞춰 넣어보기 화이팅~!!