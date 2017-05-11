from agent import agent
import numpy as np
from numba import jit # numba library 를 이용해 미리 컴파일해 학습 속도를 높임
from multiprocessing import Pool   # Multiprocessing 을 위한 라이브러리 import


# Setting Hyperparameter
vec_dim = 100
n_window = 5
whole_window = n_window * 2 + 1
learning_rate = 0.025
iteration = 5  # 반복학습횟수
negative_sample = 5 # negative sample 개수
n_cores = 2  # multiprocessing 동시 실행 프로세스 수
load_model = False  # 기존 학습된 parent weight loading 여부

# Agent 클래스 불러오기
Agent = agent(learning_rate=learning_rate, n_window=n_window, vec_dim=vec_dim)

# Agent 클래스에서 초기 세팅 함수 실행
Agent.initialize()

# 기존 학습된 모델 로딩하기
if load_model :
    Agent.W = Agent.load_model()
    print("Pre-learned weight was loaded")

# Multiprocessing 을 위해 main function을 따로 정의
@jit
def main_function(j) :

    # n_window 이후부터 시작
    idx = j + n_window

    # input word 정의
    input_word = Agent.texts[idx]

    # positive sampling
    positive_sample = Agent.texts[idx - n_window: idx ] + Agent.texts[idx + 1: idx + n_window + 1]

    # negative samping
    negative_sample_idx = Agent.negative_sampling(n_samples=negative_sample)

    # training part
    Agent.train(input_word=input_word, positive_sample=positive_sample,
                   negative_sample=negative_sample_idx)

    if (j + 1) % 1000 == 0:
        print("progress : ", np.round(100 * j / (Agent.n_total_words - n_window * 2), 2))

    if np.random.rand() < 0.00001 :
        Agent.save_model()
        print("Model saved")


# 실제 학습 실행하는 파트
if __name__ == "__main__" :

    for i in range(iteration):   # Corpus 반복 횟수
        with Pool(n_cores) as pool :

            loop = Agent.n_total_words - Agent.n_window * 2
            print(loop, "loop will be done.")
            pool.map(main_function,range(loop))
            pool.close()
            pool.join()

    # Save model Agent.W to csv
    Agent.save_model()
    print("The model finally saved\n Quit process")
