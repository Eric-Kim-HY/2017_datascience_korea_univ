from agent import agent
import numpy as np

# Setting Hyperparameter
vec_dim = 100
n_window = 5
whole_window = n_window * 2 + 1
learning_rate = 0.01
iteration = 1  # 반복학습횟수
negative_sample = 20 # negative sample 개수


if __name__ == "__main__" :

    Agent = agent(learning_rate=learning_rate, n_window = n_window, vec_dim=vec_dim)
    Agent.initialize()



    for i in range(iteration):   # Corpus 반복 횟수
        for j in range(Agent.n_total_words - n_window*2) :    # Corpus 1회 반복, 앞뒤 5개 단어씩 제외

            # n_window 이후부터 시작
            idx = j + n_window

            # input word 정의
            input_word = Agent.texts[idx]
            # positive sampling
            positive_sample = Agent.texts[idx - n_window : idx -1] + Agent.texts[idx + 1 : idx + n_window ]
            # negative samping
            negative_sample_idx = Agent.negative_sampling(n_samples=negative_sample)
            Agent.train(input_word=input_word, positive_sample=positive_sample,
                        negative_sample=negative_sample_idx, learning_rate=learning_rate)

            if j % 1000 == 0 :
                print("iter : %d progress :%f "%(i+1, np.round(100*j/(Agent.n_total_words- n_window*2),2)))

    # Save model Agent.W to csv
    Agent.save_model()