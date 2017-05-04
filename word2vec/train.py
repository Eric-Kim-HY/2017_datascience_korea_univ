from data_proc import dataprocessing
from agent import agent

# Setting Hyperparameter
vec_dim = 200
n_window = 5
whole_window = n_window * 2 + 1
learning_rate = 1e-1
train = 3  # 반복학습횟수


if __name__ == "__main__" :

    Agent = agent(learning_rate=learning_rate, n_window = n_window, vec_dim=vec_dim)
    Agent.initialize()

    dataproc = dataprocessing(vec_dim)


    for i in range(train):   # Corpus 반복 횟수
        for j in range(agent.n_total_words - n_window*2) :    # Corpus 1회 반복, 앞뒤 5개 단어씩 제외

            # n_window 이후부터 시작
            idx = j + n_window

            # input word 정의
            input_word = agent.texts[idx]

            # positive sampling
            positive_sample = agent.texts[idx - n_window : idx -1] + agent.texts[idx + 1 : idx + n_window ]

            # negative samping
            negative_sample_idx = dataproc.negative_sampling(n_samples=agent.n_words)

            Agent.train(input_word = input_word, positive_sample = positive_sample,
                        negative_sample = negative_sample_idx, learning_rate = learning_rate)

    # Save model Agent.W to csv
    Agent.W.to_csv('./wordvector.csv')