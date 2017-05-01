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
        for j in agent.n_words :    # Corpus 1회 반복

            # input word 정의
            input_word = agent.texts[j]

            # positive sampling

            #TODO j 가 1~ window-1,   -window ~ end 까지의 예외 처리하고
            # positive sampling 단어 들고옴
            positive_sample = None

            # negative samping
            negative_sample_idx = dataproc.negative_sampling(n_samples=agent.n_sample)

            Agent.train(input_word = input_word, positive_sample = positive_sample,
                        negative_sample = negative_sample_idx, learning_rate = learning_rate)

    # Save model Agent.W to csv